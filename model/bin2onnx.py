import torch
from transformers import AutoTokenizer
# from model.decoder import SketchDecoder
import torch.nn as nn
from einops import rearrange
from layers.transformer import *
from layers.improved_transformer import *


# import onnx

cfg = {
    'pix_len': 512,
    'text_len': 50,

    'tokenizer_name': 'google/bert_uncased_L-12_H-512_A-8',
    'word_emb_path': '/home/stone/Desktop/AnyFont/IconShop/ckpts/word_embedding_512.pt',
    'pos_emb_path': None,
}

# device = torch.device("cuda:0")
device = torch.device("cpu")


# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(cfg['tokenizer_name'])

"""
0: SVG END
1: MASK
2: EOM
3: FACE_END
4: LOOP END
5: CMD  END
"""
NUM_MASK_AND_EOM = 2
MASK = 1
EOM = 2
NUM_END_TOKEN = 3
CAUSAL_PAD = 3
PIX_PAD = NUM_END_TOKEN + NUM_MASK_AND_EOM
COORD_PAD = NUM_END_TOKEN + NUM_MASK_AND_EOM
SVG_END = 1
CMD_END = 2 + NUM_MASK_AND_EOM
LOOP_END = 1 + NUM_MASK_AND_EOM
FACE_END = 0 + NUM_MASK_AND_EOM
CMD_END_P = [CMD_END, CMD_END]
LOOP_END_P = [LOOP_END, LOOP_END]
FACE_END_P = [FACE_END, FACE_END]

# FIGR-SVG-svgo
BBOX = 200

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model, padding_idx=None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx)
    def forward(self, x):
        return self.embed(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=250):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(0, max_len, dtype=torch.long).unsqueeze(1)
        self.register_buffer('position', position)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self._init_embeddings()

    def _init_embeddings(self):
        nn.init.kaiming_normal_(self.pos_embed.weight, mode="fan_in")

    def forward(self, x):
        pos = self.position[:x.size(0)]
        x = x + self.pos_embed(pos)
        return self.dropout(x)



class SketchDecoder(nn.Module):
  """
  Autoregressive generative model 
  """

  def __init__(self,
               config,
               pix_len,
               text_len,
               num_text_token,
               word_emb_path=None,
               pos_emb_path=None):
    """
    Initializes FaceModel.
    """
    super(SketchDecoder, self).__init__()
    self.pix_len = pix_len
    self.embed_dim = config['embed_dim']

    self.text_len = text_len
    self.num_text_token = num_text_token
    self.num_image_token = BBOX * BBOX + PIX_PAD + SVG_END
    self.total_token = num_text_token + self.num_image_token
    self.total_seq_len = text_len + pix_len
    self.loss_img_weight = 7

    seq_range = torch.arange(self.total_seq_len)
    logits_range = torch.arange(self.total_token)

    seq_range = rearrange(seq_range, 'n -> () n ()')
    logits_range = rearrange(logits_range, 'd -> () () d')

    logits_mask = (
        ((seq_range >= text_len) & (logits_range < num_text_token)) |
        ((seq_range < text_len) & (logits_range >= num_text_token))
    )

    self.register_buffer('logits_mask', logits_mask, persistent=False)
    # Sketch encoders
    self.coord_embed_x = Embedder(BBOX+COORD_PAD+SVG_END, self.embed_dim, padding_idx=MASK)
    self.coord_embed_y = Embedder(BBOX+COORD_PAD+SVG_END, self.embed_dim, padding_idx=MASK)

    self.pixel_embed = Embedder(self.num_image_token, self.embed_dim, padding_idx=MASK)
    self.pos_embed = PositionalEncoding(max_len=self.total_seq_len, d_model=self.embed_dim)
    self.logit_fc = nn.Linear(self.embed_dim, self.total_token)
    
    decoder_layers = TransformerDecoderLayerImproved(d_model=self.embed_dim, 
                        dim_feedforward= config['hidden_dim'], nhead=config['num_heads'], dropout=config['dropout_rate'])
    decoder_norm = LayerNorm(self.embed_dim)
    self.decoder = TransformerDecoder(decoder_layers, config['num_layers'], decoder_norm)

    assert word_emb_path is not None, 'text_emb_dir must be provided'
    if word_emb_path is not None:
      self.text_emb = nn.Embedding.from_pretrained(torch.load(word_emb_path, map_location='cpu'))
      assert self.embed_dim == self.text_emb.weight.shape[1], 'emb_dim must match pretrained text embedded dim'
    if pos_emb_path is not None:
      self.text_pos_emb = nn.Embedding.from_pretrained(torch.load(pos_emb_path, map_location='cpu'))
      assert self.embed_dim == self.text_pos_emb.weight.shape[1], 'emb_dim must match pretrained text embedded dim'
  
#   @torch.jit.script
  def forward(self, pix, xy, text, sign, return_loss=False):
    '''
    pix.shape  [batch_size, max_len]
    xy.shape   [batch_size, max_len, 2]
    mask.shape [batch_size, max_len]
    text.shape [batch_size, text_len]
    '''
    pixel_v = pix
    xy_v = xy
    # pixel_mask = mask[:, :-1] if return_loss else mask

    c_bs, c_seqlen, device = text.shape[0], text.shape[1], text.device
    # if pixel_v[0] is not None:
    # if not (pixel_v == 0).all():
    # if not k == text.shape[1]:
    if sign != 0:
      c_seqlen += pixel_v.shape[1]  
    # c_seqlen += torch.where(not k == text.shape[1], pix.shape[1], torch.tensor(0))

    # Context embedding values
    context_embedding = torch.zeros((1, c_bs, self.embed_dim)).to(device) # [1, bs, dim]

    # tokens.shape [batch_size, text_len, emb_dim]
    tokens = self.text_emb(text)
    # print('tokens', tokens) # device='cuda:0'

    # Data input embedding
    # if pixel_v[0] is not None:
    # if not (pixel_v == 0).all():
    # if not k == text.shape[1]:
    if sign != 0:
      # coord_embed.shape [batch_size, max_len-1, emb_dim]
      # pixel_embed.shape [batch_size, max_len-1, emb_dim] 
    #   print('xy_v', xy_v)
    #   print('xy_v0', xy_v[...,0])
    #   print('xy_v1', xy_v[...,1])
      coord_embed = self.coord_embed_x(xy_v[...,0]) + self.coord_embed_y(xy_v[...,1]) # [bs, vlen, dim]
    #   print('coord_embed', coord_embed)
      pixel_embed = self.pixel_embed(pixel_v)
    #   print('pixel_embed', pixel_embed)
      embed_inputs = pixel_embed + coord_embed
    #   print('embed_inputs', embed_inputs)

      # tokens.shape [batch_size, text_len+max_len-1, emb_dim]
      tokens = torch.cat((tokens, embed_inputs), dim=1)
    #   print('tokens', tokens)
    
    # coord_embed = torch.where(not k == text.shape[1], self.coord_embed_x(xy[...,0]) + self.coord_embed_y(xy[...,1]), torch.zeros_like(xy))
    # pixel_embed = torch.where(not k == text.shape[1], self.pixel_embed(pix), torch.zeros_like(pix))
    # embed_inputs = torch.where(not k == text.shape[1],pixel_embed + coord_embed, torch.zeros_like(xy))
    # tokens = torch.where(not k == text.shape[1], torch.cat((tokens, embed_inputs), dim=1), tokens)

    # embeddings.shape [text_len+1 or text_len+max_len, batch_size, emb_dim]
    embeddings = torch.cat([context_embedding, tokens.transpose(0,1)], axis=0)
    # print('embeddings', embeddings)
    decoder_inputs = self.pos_embed(embeddings) 
    # print('decoder_inputs', decoder_inputs)

    memory_encode = torch.zeros((1, c_bs, self.embed_dim)).to(device)
    
    # nopeak_mask.shape [c_seqlen+1, c_seqlen+1]
    nopeak_mask = torch.nn.Transformer.generate_square_subsequent_mask(c_seqlen+1).to(device)  # masked with -inf
    # if pixel_mask is not None:
    #   # pixel_mask.shape [batch_size, text_len+max_len]
    #   pixel_mask = torch.cat([(torch.zeros([c_bs, context_embedding.shape[0]+self.text_len])==1).to(device), pixel_mask], axis=1)  
    decoder_out = self.decoder(tgt=decoder_inputs, memory=memory_encode, memory_key_padding_mask=None,
                               tgt_mask=nopeak_mask, tgt_key_padding_mask=None)

    # Logits fc
    logits = self.logit_fc(decoder_out)  # [seqlen, bs, dim] 
    logits = logits.transpose(1,0)  # [bs, textlen+seqlen, total_token] 

    logits_mask = self.logits_mask[:, :c_seqlen+1]
    max_neg_value = -torch.finfo(logits.dtype).max
    logits.masked_fill_(logits_mask, max_neg_value)

    # if return_loss:
    #   logits = rearrange(logits, 'b n c -> b c n')
    #   text_logits = logits[:, :, :self.text_len]
    #   pix_logits = logits[:, :, self.text_len:]

    #   pix_logits = rearrange(pix_logits, 'b c n -> (b n) c')
    #   pix_mask = ~mask.reshape(-1)
    #   pix_target = pix.reshape(-1) + self.num_text_token

    #   text_loss = F.cross_entropy(text_logits, text)
    #   pix_loss = F.cross_entropy(pix_logits[pix_mask], pix_target[pix_mask], ignore_index=MASK+self.num_text_token)
    #   loss = (text_loss + self.loss_img_weight * pix_loss) / (self.loss_img_weight + 1)
    #   return loss, pix_loss, text_loss
    return logits


# 加载 PyTorch 模型
sketch_decoder = SketchDecoder(
    config={
        'hidden_dim': 1024,
        'embed_dim': 512,
        'num_layers': 16,
        'num_heads': 8,
        'dropout_rate': 0.1
    },
    pix_len=cfg['pix_len'],
    text_len=cfg['text_len'],
    num_text_token=tokenizer.vocab_size,
    word_emb_path=cfg['word_emb_path'],
    pos_emb_path=cfg['pos_emb_path'],
)

# 加载预训练权重
sketch_decoder.load_state_dict(torch.load("/home/stone/Desktop/AnyFont/IconShop/proj_log/FIGR_SVG/epoch_100/pytorch_model.bin"))

# 设置为评估模式
sketch_decoder.to(device).eval()

n_samples =4

# # 定义示例输入 for 初始化
# sign = 0
# pixel_seq = torch.randint(0, BBOX+COORD_PAD+SVG_END, (n_samples, 1), device=device)
# xy_seq = torch.randint(0, BBOX+COORD_PAD+SVG_END, (n_samples, 1, 2), device=device)

sign = 1
# 定义示例输入
pixel_seq = torch.randint(0, BBOX+COORD_PAD+SVG_END, (n_samples, 2), device=device)
xy_seq = torch.randint(0, BBOX+COORD_PAD+SVG_END, (n_samples, 2, 2), device=device)

# 直接赋值给 pixel_seq
# pixel_seq = torch.tensor([
#     [3, 5668],
#     [3, 8275],
#     [3, 8072],
#     [3, 8276],
#     [3, 6873],
#     [3, 6267],
#     [3, 6268],
#     [3, 3271]
# ], device=device)

# # 直接赋值给 xy_seq
# xy_seq = torch.tensor([
#     [[3, 3], [68, 34]],
#     [[3, 3], [75, 47]],
#     [[3, 3], [72, 46]],
#     [[3, 3], [76, 47]],
#     [[3, 3], [73, 40]],
#     [[3, 3], [67, 37]],
#     [[3, 3], [68, 37]],
#     [[3, 3], [71, 22]]
# ], device=device)
text = torch.randint(0, tokenizer.vocab_size, (n_samples, cfg['text_len']), device=device) # int64

# scripted_SketchDecoder = torch.jit.script(sketch_decoder)

# print(scripted_SketchDecoder.code)


# Convert to ScriptModule using torch.jit.script
scripted_model = torch.jit.script(sketch_decoder(pixel_seq, xy_seq, text, sign))


input_dict = (pixel_seq, xy_seq, text, sign)


# 转换为 ONNX
torch.onnx.export(
    scripted_model,
    input_dict,
    # f"iconshop_{sign}.onnx",
    "iconshop_scripted.onnx",
    input_names=['pixel_seq', 'xy_seq','text', 'sign'],
    output_names=['output'],
    dynamic_axes={'pixel_seq': {0: 'batch_size',
                                1: 'max_len'},
                  'xy_seq': {0: 'batch_size',
                             1: 'max_len'},
                  'text': {0: 'batch_size'}, 
                  'output': {0: 'batch_size'}},
    opset_version=14
)

print("模型成功转换为 ONNX 格式")
