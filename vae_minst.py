import torch
import torchvision
import torch.utils.data
import torch.nn

transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

train_set=torchvision.datasets.MNIST(
    root='./minst_data/',
    train=False,
    transform=transform,
    download=True
)

train_set_loader=torch.utils.data.DataLoader(
    dataset=train_set,
    shuffle=True,
    batch_size=64
)

class VAE(torch.nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.encoder =torch.nn.Sequential(
        torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),#14*14
        torch.nn.BatchNorm2d(64),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),#7*7
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),

        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),#7*7
        torch.nn.BatchNorm2d(128),
        torch.nn.LeakyReLU(0.2, inplace=True),
        )

        self.get_mu=torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 32)
        )
        self.get_logvar = torch.nn.Sequential(
            torch.nn.Linear(128 * 7 * 7, 32)
        )
        self.get_temp = torch.nn.Sequential(
            torch.nn.Linear(32, 128 * 7 * 7)
        )
        self.decoder = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        torch.nn.ReLU(inplace=True),

        torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
        torch.nn.Sigmoid()
        )



    def get_z(self,mu,logvar):
        eps=torch.randn(mu.size(0),mu.size(1))#64,32
        eps=torch.autograd.Variable(eps).cuda()
        z=mu+eps*torch.exp(logvar/2)
        return z


    def forward(self, x):
        out1=self.encoder(x)
        mu=self.get_mu(out1.view(out1.size(0),-1))#64,128*7*7->64,32
        out2=self.encoder(x)
        logvar=self.get_logvar(out2.view(out2.size(0),-1))

        z=self.get_z(mu,logvar)
        out3=self.get_temp(z).view(z.size(0),128,7,7)

        return self.decoder(out3),mu,logvar

def loss_fun(new_x,old_x,mu,logvar):
    BCE=torch.nn.functional.binary_cross_entropy(new_x,old_x,size_average=False)
    KLD=-0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE+KLD

vae=VAE().cuda()

optimizer=torch.optim.Adam(vae.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-08,weight_decay=0)

epoch_n=15
total_loss = 0
for epoch in range(epoch_n):
    for i,(data,_) in enumerate(train_set_loader):
        old_img=torch.autograd.Variable(data).cuda()
        new_img,mu,logvar=vae.forward(old_img)
        loss=loss_fun(new_img,old_img,mu,logvar)

        total_loss+=loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            sample = torch.autograd.Variable(torch.randn(64, 32)).cuda()
            sample = vae.decoder(vae.get_temp(sample).view(64, 128, 7, 7)).cpu()
            torchvision.utils.save_image(sample.data.view(64, 1, 28, 28), './result_vae_minst/sample_' + str(epoch) + '.png')

            print('Train Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
            epoch, i * len(data), len(train_set_loader.dataset),
            100.* i / len(train_set_loader), loss.data.item() / len(data)))


    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, total_loss / len(train_set_loader.dataset)))



