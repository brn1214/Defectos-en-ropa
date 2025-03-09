from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

# Mover el modelo a la GPU si está disponible
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Poner el modelo en modo de evaluación  
model.eval()

# Inicializar el métrico mAP
metric = MeanAveragePrecision()

# Listas para almacenar las etiquetas predichas y reales
predicciones = [] 

with torch.no_grad():
    for images in test_loader:
        
        images = list(image.to(device) for image in images) 

        # Forward pass 
        predictions = model(images)

        # Actualizar el métrico mAP con las predicciones y los targets
        metric.update(predictions, targets)

        # Almacenar predicciones y targets para calcular Precision, Recall y F1-Score
        for p in predictions:
          predicciones.append(p)
        

# Calcular el mAP
result = metric.compute()
 
# Calcular Precision, Recall y F1-Score
precision = precision_score(targets, predicciones, average='weighted')
recall = recall_score(targets, predicciones, average='weighted')
f1 = f1_score(targets, predicciones, average='weighted')

# Imprimir los resultados del mAP
print(f'mAP: {result["map"]}')  # mAP promedio 
print(f'mAP_75: {result["map_75"]}')  # mAP con un IoU de 0.75 

# Imprimir las métricas de Precision, Recall y F1-Score
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
 
