import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self,input):
        return input.view(input.size(0),-1)

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0),2560,1,1)


class VAE(nn.Module):
    def __init__(self,img_channels=3,h_dim=2560,stride=2,z_dim=200,dropout=0.2,relu=0.1):
        super(VAE,self).__init__()
	
	if relu == 0:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU(relu)

        self.encoder=nn.Sequential(
        '''nn.Conv2d(1,16,kernel_size=(1,40),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(16,32,kernel_size=(10,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(32,64,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,256,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        
        
        Flatten(),
        # nn.Dropout(0.2)'''
	nn.Conv2d(1, 64, kernel_size=(1,40), stride=1, padding=0, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(64),
        activation
        nn.Conv2d(64, 128, kernel_size=(5,1), stride=1, padding=0, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(128),
        activation
	nn.Conv2d(128, 256, kernel_size=(3,1), stride=1, padding=0, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(256),
        activation
        '''nn.Linear(1024, 512, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm1d(512),
        activation'''
        )
        
        self.fc1=nn.Linear(h_dim,z_dim)
	nn.Dropout2d(dropout),
        nn.BatchNorm1d(z_dim),
        activation
        self.fc2=nn.Linear(h_dim,z_dim)
	nn.Dropout2d(dropout),
        nn.BatchNorm1d(z_dim),
        activation
        self.fc3=nn.Linear(z_dim,h_dim)
	nn.Dropout2d(dropout),
        nn.BatchNorm1d(h_dim),
        activation
        print(h_dim)
        print(z_dim)
        
        
        self.decoder=nn.Sequential(
        #UnFlatten(),  
        '''nn.ConvTranspose2d(h_dim,256,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256,128,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128,64,kernel_size=(5,1),stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,64,kernel_size=(5,1),stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,32,kernel_size=(5,1),stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32,16,kernel_size=(10,1),stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16,1,kernel_size=(1,40),stride=1),
        nn.ReLU(inplace=True)'''
	nn.ConvTranspose2d(h_dim, 256, kernel_size=(3,1), stride=1, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(128),
        activation
	nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=1, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(128),
        activation
  	nn.ConvTranspose2d(128, 64, kernel_size=(5,1), stride=1, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(64),
        activation
	nn.ConvTranspose2d(64, 1, kernel_size=(1,40), stride=1, bias=False),
        nn.Dropout2d(dropout),
        nn.BatchNorm2d(1),
        activation


        #Flatten(),
        #nn.Linear(121,20*130),
        #nn.Sigmoid()
               
        )
    def reparameterize(self,mu,log_var):
        std=torch.exp(0.5*log_var)
        eps=torch.randn_like(std)
        z=mu+eps*std
        return z
    def forward(self,x):
        h=self.encoder(x)
        
        mu,log_var=self.fc1(h),self.fc2(h)
        z=self.reparameterize(mu,log_var)
        z=self.fc3(z)
        

        return self.decoder(z),mu,log_var

'''class MMDVAE(nn.Module):
    def __init__(self,img_channels=3,h_dim=2304,stride=2,z_dim=200):
        super(MMDVAE,self).__init__()
	
        #self.features=nn.Sequential(nn.Conv2d(3,32,kernel_size=3),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2),nn.Dropout(0.3),nn.BatchNorm2d(32),nn.Conv2d(32,64,kernel_size=3),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2),nn.Dropout(0.3),nn.BatchNorm2d(64)
        #	)
        self.encoder=nn.Sequential(
        nn.Conv2d(1,16,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.Conv2d(16,32,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(32,64,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,256,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        Flatten(),
        # nn.Dropout(0.2)
        )
        self.fc1=nn.Linear(h_dim,z_dim)
        self.fc2=nn.Linear(z_dim,h_dim)
        #self.fc3=nn.Linear(z_dim,h_dim)
        
        self.decoder=nn.Sequential(
        UnFlatten(),  
        nn.ConvTranspose2d(h_dim,256,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256,128,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128,64,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,64,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,32,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32,16,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16,1,kernel_size=3,stride=2),
        nn.ReLU(inplace=True)


        #Flatten(),
        #nn.Linear(121,20*130),
        #nn.Sigmoid()
               
        )
    
    def forward(self,x):
        h=self.encoder(x)
        z = self.fc1(h)
        out = self.fc2(z)
        return self.decoder(out), z

class AE(nn.Module):
    def __init__(self,img_channels=3,h_dim=50176,stride=2,z_dim=2000):
        super(AE,self).__init__()
    
        #self.features=nn.Sequential(nn.Conv2d(3,32,kernel_size=3),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2),nn.Dropout(0.3),nn.BatchNorm2d(32),nn.Conv2d(32,64,kernel_size=3),nn.ReLU(inplace=True),nn.MaxPool2d(kernel_size=2),nn.Dropout(0.3),nn.BatchNorm2d(64)
        #   )
        self.encoder=nn.Sequential(
        nn.Conv2d(1,16,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.Conv2d(16,32,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.Conv2d(32,64,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,64,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64,128,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128,256,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        Flatten(),
        # nn.Dropout(0.2)
        )
        self.fc1=nn.Linear(h_dim,z_dim)
        self.fc2=nn.Linear(z_dim,h_dim)
        #self.fc3=nn.Linear(z_dim,h_dim)
        
        self.decoder=nn.Sequential(
        UnFlatten(),  
        nn.ConvTranspose2d(h_dim,256,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(256),
        nn.ConvTranspose2d(256,128,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.ConvTranspose2d(128,64,kernel_size=3,stride=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,64,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64,32,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32,16,kernel_size=3,stride=2),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16,1,kernel_size=3,stride=2),
        nn.ReLU(inplace=True)


        #Flatten(),
        #nn.Linear(121,20*130),
        #nn.Sigmoid()
               
        )
    
    def forward(self,x):
        h=self.encoder(x)
        '''
        '''z = self.fc1(h)
        out = self.fc2(z)
     
        #return self.decoder(h)

class Encoder(nn.Module):
    def __init__(self,  Z_DIM, log_var_=None, dropout=0.2, relu=0.1, n_filters=40):
        super(Encoder, self).__init__()
      
        kernel_size = (height, width)
        stride = (heigh, width)
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Fc1: Units = 512, tanh, BatchNorm
        Gauss: Units = 128
        
        1*(19*40) --> 64*(19*1) --> 128*(9*1) --> 256*(4*1) --> (512) --> (128)
       
        if relu == 0:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU(relu)

        self.F = n_filters

        self.Conv_1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,self.F), stride=1, padding=0, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(64),
            activation
        )
        self.Conv_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,1), stride=(2,1), padding=0, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(128),
            activation
        )
        self.Conv_3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,1), stride=(2,1), padding=0, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(256),
            activation
        )

        self.Fc1 =nn.Sequential(
            nn.Linear(1024, 512, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm1d(512),
            activation
        )

        self.mu = nn.Linear(512, Z_DIM, bias=False)
        self.fix = log_var_
        if log_var_ != None:
            self.log_var = log_var_
        else:
            self.log_var = nn.Linear(512, Z_DIM, bias=False)

    def forward(self, input):
        output = input.view(-1, 1, 19, self.F)
        out = self.Conv_1(output)
        out = self.Conv_2(out)
        out = self.Conv_3(out)
        out = out.view(-1, 1024)
        out = self.Fc1(out)
        mu = self.mu(out)
        if self.fix == None:
            log_var = self.log_var(out)
        else:
            log_var = Variable(2*torch.log(self.log_var * torch.ones(mu.size())))
        if torch.cuda.is_available():
            log_var = log_var.cuda()
        return mu, log_var

class Decoder(nn.Module):
    def __init__(self, Z_DIM, dropout=0.2, relu=0.1, n_filters = 40):
        super(Decoder, self).__init__()
        
        Gauss: Units = 48
        Fc1: Units = 512, tanh, BatchNorm
        Cov3: filters = 256, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Cov2: filters = 128, filter size = 3 * 1, stride = (2,1), tanh(), BatchNorm
        Conv1: filters = 64, filter size = 1 * F, stride = (1,1), tanh(), BatchNorm
        1*(128*1) --> 1*(512*1) --> 256*(4*1) --> 128*(9*1) --> 64*(19*1) --> 1*(19*40)
        
        if relu == 0:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU(relu)


        self.F = n_filters

        self.Gauss = nn.Linear(Z_DIM, 512, bias=False)

        self.Fc1 = nn.Sequential(
            nn.Linear(512, 1024, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm1d(1024),
            activation
        )
        #self.deconv3 = nn.ConvTranspose2d(4*DIM, 2*DIM, kernel_size=(3,1),stride=(2,1))

        self.Conv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=(3,1), stride=(2,1), bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(128),
            activation
        )

        self.Conv_2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=(3,1), stride=(2,1), bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(64),
            activation
        )


        self.Conv_1 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=(1,self.F), stride=1, bias=False),
            nn.Dropout2d(dropout),
            nn.BatchNorm2d(1),
            activation
        )

    def forward(self, input):
        output = self.Gauss(input)
        output = self.Fc1(output)
        output = output.view(-1, 256, 4, 1)
        output = self.Conv_3(output)
        output = self.Conv_2(output)
        output = self.Conv_1(output)
        output = output.view(-1, 19*self.F)
        return output

class VAE2(nn.Module):
    def __init__(self, Z_DIM, log_var_=None, dropout=0.2, relu=0.1, n_filters=40):
        super(VAE2, self).__init__()

       
        VAE:
        Input --> Encoder --> mu,log_var --> Decoder --> Output
        1 * (9*40) -->  Encoder --> 1 * 48 --> Decoder --> 1 * (9*40) 
        
        self.encoder = Encoder(Z_DIM, log_var_, dropout=dropout, relu=relu, n_filters=n_filters)
        self.decoder = Decoder(Z_DIM, dropout=dropout, relu=relu, n_filters=n_filters)

    def forward(self, input):
        mu, log_var = self.encoder(input)
        # reparam
        z = self.reparameterize(mu.float(), log_var.float())
        output = self.decoder(z)
        return output, mu, log_var

    def reparameterize(self, mu, log_var):
        if self.training:
            std = log_var.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            if torch.cuda.is_available():
                eps = eps.cuda()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def sample(self, z):
        return self.decoder(z)'''



