# Verificar si se puede trabajar con CUDA
device = torch.device('cuda') if torch.cuda.is_available() 
                               else torch.device('cpu')

model.to(device)

# Definir el optimizador
optimizer = torch.optim.SGD(params, lr=0.01)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0

   # Training loop
    for images, targets in train_loader:
      
        # Las imagenes y etiquetas se mueven al GPU o CPU para que el modelo haga las operaciones matriciales
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
        optimizer.zero_grad()

        # Forward
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward 
        losses.backward()
        optimizer.step()
        train_loss += losses.item()
 
    print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader)}')
print("Training complete!")
