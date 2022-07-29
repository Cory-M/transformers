import torch
import pdb
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from simple_config import SimpleConfig

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased-finetuned-sst-2-english', 
            torchscript=True) # setting torchscript=True automatically sets 'return_dict'=False
model.eval()

def overwrite():
    model.config = SimpleConfig()
    model.distilbert.config = SimpleConfig()
    model.distilbert.transformer.config = SimpleConfig()

overwrite()

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs)


print('origin prediction:', logits)
predicted_class_id = logits.argmax().item()

scr = torch.jit.script(model)
print('scripted model prediction:', scr(**inputs))

#torch.jit.save(scr, 'out.pt')
print('successfully scripted and save to out.pt')









