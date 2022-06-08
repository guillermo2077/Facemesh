el .py que se debe ejecutar para usar el programa es FaceMesh.py, 
el otro archivo de python se encarga de generar el archivo exp_classifier.h5,
por lo tanto si este ultimo archivo ya se encuentra dentro de assets, 
no es necesario ejecutar Exp_classifier.py.

Para evitar demasiada carga el proceso de clasificación de imagen solo
tiene lugar cada 3 segundos en FaceMesh.py, entonces cuando se pruebe el
programa se recomienda mantener una misma expresión durante al menos 3 segundos
siempre.