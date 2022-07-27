#from model.clip_transformer import CLIPTransformer
import torch
import torch.nn as nn

from transformers import CLIPModel
from simple_config import SimpleConfig

import pdb

torch.manual_seed(7)


# set up the Model to call get_image_features/get_text_features separately
# calling scripte_model.get_image_features is not supported
class Model(nn.Module):
    def __init__(self, torchscript=True):
        super(Model, self).__init__()
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", torchscript=True)
    def forward(self, video, input_ids, attention_mask):
        video_out = self.model.get_image_features(video)
        text_out = self.model.get_text_features(input_ids, attention_mask)
        return video_out, text_out
    
model = Model(torchscript=True)

def overwrite():
    model.model.text_model.config = SimpleConfig()
    model.model.text_model.encoder.config = SimpleConfig()
    model.model.vision_model.config = SimpleConfig()
    model.model.vision_model.encoder.config = SimpleConfig()
    
    model.model.config = SimpleConfig()

def generate_data(bs=3, num_frame=12, H=224, W=224, token_len=9, cuda=False):
    video =  torch.randn((bs, 3, H, W)) # image shape
    input_ids =  torch.randint(0, 1000, (bs, token_len), dtype=torch.int64)
    attention_mask = torch.randint(0, 1, (bs, token_len), dtype=torch.int64)

    if cuda:
        video = video.cuda()
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    data = {'video': video,
        'input_ids': input_ids,
        'attention_mask': attention_mask}
    return data


def testing(model, script_model, cuda=True):
    data = generate_data(cuda=cuda)
    if cuda:
        model = model.cuda()
        script_model = script_model.cuda()
    
    out_a = model(data['video'], data['input_ids'], data['attention_mask'])
    out_b = script_model(data['video'], data['input_ids'], data['attention_mask'])
    assert (out_a[0] == out_b[0]).all()
    assert (out_a[1] == out_b[1]).all()
    print('pass test with cuda set {}'.format(cuda))


if __name__ == '__main__':
    # scripting the model
    model.eval()
    overwrite()
    script_model = torch.jit.script(model)
    
    # quick testing
    testing(model, script_model, cuda=False)
    testing(model, script_model, cuda=True)
    
    # save the scripted model
    torch.jit.save(script_model, 'out.pt')
    print('saving scripted model at out.pt')
