import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
import random
from tqdm import tqdm
import numpy as np

def generate_test_dataset(
    train_user, 
    train_item,
    test_user,
    test_item,
    user_ids,
    item_ids,
    test_size = 10000
):
    #pbar = tqdm(total=test_size)
    train_positves = [str(u) + "|" + str(i) for u, i in zip(train_user, train_item)]
    test_positves = [str(u) + "|" + str(i) for u, i in zip(test_user, test_item)]
    
    randomly_selected_test_positves = random.sample(test_positves, test_size)
    test_positives_users = [int(p.split("|")[0]) for p in randomly_selected_test_positves]
    test_positives_items = [int(p.split("|")[1]) for p in randomly_selected_test_positves]
    
    test_negatives_users = []
    test_negatives_items = []
    while len(test_negatives_users) < test_size:
        random_user = random.choice(user_ids)
        random_item = random.choice(item_ids)
        random_pair = str(random_user) + "|" + str(random_item)
        if random_pair not in train_positves and random_pair not in test_positves:
            test_negatives_users.append(int(random_pair.split("|")[0]))
            test_negatives_items.append(int(random_pair.split("|")[1]))
            #pbar.update(1)
            
    user_id = np.concatenate([
        np.expand_dims(np.array(test_positives_users), axis = 1),
        np.expand_dims(np.array(test_negatives_users), axis = 1)
    ], axis = 0)
    item_id = np.concatenate([
        np.expand_dims(np.array(test_positives_items), axis = 1),
        np.expand_dims(np.array(test_negatives_items), axis = 1)
    ], axis = 0)
    return user_id, item_id, [1]*test_size + [0]*test_size

def draw_roc_curve(y_true, y_score):
	fpr, tpr, thresholds = roc_curve(y_true, y_score)
	roc_auc = auc(fpr, tpr)

	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic')
	plt.legend(loc="lower right")
	plt.show()

	return fpr, tpr, thresholds

def draw_confusion_matrix(y_true, y_pred):
	cm = confusion_matrix(y_true, y_pred)
	tn, fp, fn, tp = cm.ravel()
	title = 'Confusion Matrix'

	fig, ax = plt.subplots()
	im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
	ax.figure.colorbar(im, ax=ax)
	ax.set(xticks=[0, 1],
	       yticks=[0, 1],
	       xticklabels=['Negative', 'Positive'],
	       yticklabels=['Negative', 'Positive'],
	       title=title,
	       ylabel='True label',
	       xlabel='Predicted label')

	# Loop over data dimensions and create text annotations.
	thresh = cm.max() / 2.
	for i in range(cm.shape[0]):
	    for j in range(cm.shape[1]):
	        ax.text(j, i, format(cm[i, j], 'd'),
	                ha="center", va="center",
	                color="white" if cm[i, j] > thresh else "black")
	fig.tight_layout()
	plt.show()