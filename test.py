import torch
import torchvision
from loader import get_loader
from loader import chew

impath = "/Users/seanmorrison/Desktop/CelebA/img_align_celeba/"
attpath = "/Users/seanmorrison/Desktop/CelebA/list_attr_celeba.txt"
attributes = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']

"5_o_Clock_Shadow"
"Arched_Eyebrows"
"Attractive" 
"Bags_Under_Eyes" 

"Bangs"
"Big_Lips" 
"Big_Nose"

"Bald"
"Black_Hair" 
"Blond_Hair"
"Brown_Hair"
"Gray_Hair"

"Blurry"
 
"Bushy_Eyebrows" 
"Chubby" 
"Double_Chin" 
"Eyeglasses"
"Goatee"
 
"Heavy_Makeup" 
"High_Cheekbones" 
"Male"
"Mouth_Slightly_Open" 
"Mustache"
"Narrow_Eyes" 
"No_Beard"
"Oval_Face" 
"Pale_Skin"
"Pointy_Nose" 
"Receding_Hairline" 
"Rosy_Cheeks"
"Sideburns"
"Smiling"
"Straight_Hair" 
"Wavy_Hair"
"Wearing_Earrings" 
"Wearing_Hat"
"Wearing_Lipstick" 
"Wearing_Necklace"
"Wearing_Necktie"
"Young"

dset = get_loader(impath, attpath, attributes, batch_size=16, image_size=200)
xs, ys = next(iter(dset))
print(ys)
print(xs)
print(chew(ys))
print(xs.size())
print(ys.size())