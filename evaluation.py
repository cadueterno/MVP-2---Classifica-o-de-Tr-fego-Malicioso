from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

class Evaluator:
    @staticmethod
    def evaluate(model, X_test, y_test):
        y_pred = model.predict(X_test)
        print('=== Classification Report ===')
        print(classification_report(y_test, y_pred))
        print('=== Confusion Matrix ===')
        print(confusion_matrix(y_test, y_pred))
        try:
            if hasattr(model.grid.best_estimator_['clf'], 'predict_proba'):
                print('ROC-AUC Score:', roc_auc_score(y_test, model.grid.predict_proba(X_test)[:,1]))
        except:
            pass
