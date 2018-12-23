from folder import ImageFolder
from sklearn import svm
from torchvision import transforms
import torch
from model import Extractor
from tqdm import tqdm

transforms = transforms.Compose([transforms.Resize((224,224)),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229,0.224,0.225])])
def train(loader, model, clf):
    print("============>training<============")
    test_dataset = ImageFolder(root="./data/icdar2015/test_patches",
                               transform=transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)
    pbar = tqdm(total=len(loader))
    for cnt, data in enumerate(loader):
        pbar.update(1)
        im, ann, im_fns = data[0], data[1], data[2]
        im = im.cuda()
        feat = model(im)
        x = feat.cpu().numpy()
        y = ann.numpy()
        try:
            clf.fit(x, y)
        except ValueError:
            continue

    pbar.close()
    print("\n")
    print("============>testing<============")
    test(test_loader, clf)

def test(loader, clf):

    pbar = tqdm(total=len(loader))

    precision = 0.0
    for cnt, data in enumerate(loader):
        pbar.update(1)
        im, ann, im_fns = data[0], data[1], data[2]
        im = im.cuda()
        feat = model(im)
        x = feat.cpu().numpy()
        y = ann.numpy()

        precision = precision + clf.score(x, y)
    pbar.close()


    print("\nAverage Precision is {}".format(precision/len(loader)))

if __name__ == '__main__':

    icdar_patches = "./data/icdar2015/patches"

    dataset = ImageFolder(root=icdar_patches, transform=transforms)
    loader = torch.utils.data.DataLoader(dataset, batch_size=20, shuffle=True)

    # extractor for deep features
    model = Extractor()
    model = model.cuda()
    for params in model.parameters():
        params.requires_grad = False

    # vanilla svm
    clf = svm.SVC(kernel="rbf", gamma=10)

    train(loader=loader, model=model, clf=clf)
