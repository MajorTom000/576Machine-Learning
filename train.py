
from  models import *
from Dataloader import *
def train_model(module,epochs):
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(module.parameters(), lr=0.000015,weight_decay=0.0001)
  train_ls=[]
  valid_ls=[]
  train_accuracy=[]
  valid_accuracy=[]
  for epoch in range(epochs):
      correct = 0
      total = 0
      ls_loss = []
      valid_loss=[]
      for i, (images, labels) in enumerate(train_loader):
          images = images.to(device)
          labels = labels.to(device)
          torch.cuda.empty_cache()
          outputs = module(images)
          loss = criterion(outputs, labels.long())
          ls_loss.append(loss.item())
          optimizer.zero_grad()
          loss.backward()
          torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=1)
          optimizer.step()

          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels.long()).sum().item()
          predicted = predicted.cpu().numpy()
          labels = labels.cpu().numpy()


      epoch_loss = sum(ls_loss) / len(ls_loss)
      train_ls.append(epoch_loss)
      accuracy = 100 * correct / total
      train_accuracy.append(accuracy)
      print('[{}/{}] Training: loss:{:.4f}, Accuracy：{:.2f}%'.format(epoch+1,epochs,epoch_loss,accuracy))


      module.eval()
      val_accuracy = 0
      total = 0
      with torch.no_grad():

          for images, labels in valid_loader:
              images = torch.Tensor(images).float().to(device)
              labels = torch.Tensor(labels).float().to(device)
              outputs = module(images)
              _, predicted = torch.max(outputs.data, 1)
              loss = criterion(outputs, labels.long())
              valid_loss.append(loss.item())
              total += labels.size(0)
              val_accuracy += (predicted == labels.long()).sum().item()
          epoch_loss = sum(valid_loss) / len(valid_loss)
          valid_ls.append(epoch_loss)
          predicted = predicted.cpu().numpy()
          labels = labels.cpu().numpy()
          val_accuracy = 100 * val_accuracy / total

          valid_accuracy.append(val_accuracy)
          print('[{}/{}] Valid: loss:{:.4f}, Accuracy：{:.2f}% '.format(epoch+1,epochs,epoch_loss, val_accuracy))

  return module,train_ls,valid_ls,train_accuracy,valid_accuracy


epochs=20


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
module=my_fnn_model(224*224*3,32,10).to(device)
model1,ls_loss1,valid_loss1,train_accuracy1,val_accuracy1=train_model(module,epochs)



module=my_cnn_model().to(device)
model2,ls_loss2,valid_loss2,train_accuracy2,val_accuracy2=train_model(module,epochs)


hidden =[[64,32,1,1],[32,64,1,1],[64,128,1,1],[128,256,1,1]]
module = My_resnet(3, hidden, 10) .to(device)
model3,ls_loss3,valid_loss3,train_accuracy3,val_accuracy3=train_model(module,epochs)

hidden =[[64,32,1,1],[32,64,1,1],[64,128,1,1],[128,256,1,1]]
module =My_resnet(3, hidden, 10,'attention') .to(device)
model4,ls_loss4,valid_loss4,train_accuracy4,val_accuracy4=train_model(module,epochs)



