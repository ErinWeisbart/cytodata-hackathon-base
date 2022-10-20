import pandas as pd
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def eval_test_set(df_mitocells, best_clf, X_train, y_train, X_test, y_test):
    # Apply the model (to the training dataset)
    y_trainpred = best_clf.predict(X_train)
    # Apply the model (to the test dataset)
    y_testpred = best_clf.predict(X_test)

    # Show performance as a confusion matrix
    cm_train = confusion_matrix(y_train, y_trainpred, labels = df_mitocells['cell_stage'].unique())                        
    cm_test = confusion_matrix(y_test, y_testpred, labels = df_mitocells['cell_stage'].unique())                        

    # plot it
    labels = df_mitocells['cell_stage'].unique()
    cm_train_df = pd.DataFrame(cm_train)
    cm_test_df = pd.DataFrame(cm_test)
    score_test = accuracy_score(y_test,y_testpred) #compute accuracy score
    score_train = accuracy_score(y_train,y_trainpred) #compute accuracy score
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5), dpi=100)
    #train
    sns.heatmap(cm_train_df, annot=True, fmt='d',ax=axes[0])
    axes[0].set_title(f'Train accuracy is {score_train:.2f}')
    axes[0].set_ylabel('True')
    axes[0].set_xlabel('Predicted')
    axes[0].set_xticklabels([''] + labels)
    axes[0].set_yticklabels([''] + labels)
    #test
    sns.heatmap(cm_test_df, annot=True, fmt='d',ax=axes[1])
    axes[1].set_title(f'Test accuracy is {score_test:.2f}')
    axes[1].set_ylabel('True')
    axes[1].set_xlabel('Predicted')
    axes[1].set_xticklabels([''] + labels)
    axes[1].set_yticklabels([''] + labels)
    return fig, y_trainpred, y_testpred
