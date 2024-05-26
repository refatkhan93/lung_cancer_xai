import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn import svm, tree, neighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dropout, Input,  Embedding, MultiHeadAttention, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import keras
from keras.utils import to_categorical
from tensorflow.keras.models import Model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import fbeta_score, make_scorer, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import lime
import lime.lime_tabular
# import plot_roc_curve
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE
class MyUtilityFunctions:
    def __init__(self):
        print('Utility class created')

    def read_dataset(self, path, keys=None, total_rows=1001):
        dataFrame = pd.read_excel(path)
        if keys is not None:
            dataFrame = dataFrame[keys]
            dataFrame = dataFrame[:total_rows]
        return dataFrame
    
    def convert_to_float(self, dataFrame): #filter abnormal value type
        for column in dataFrame.columns:
            try:
                dataFrame[column] = pd.to_numeric(dataFrame[column], errors='coerce').astype(float)
            except:
                pass

        return dataFrame

    def filter_dataset(self, dataset): #filter abnormal high values
        for col in dataset:
            if col in ['Age', 'VitamenD(nmol/L)', 'Diagnosis']:
                continue
            for i, row in enumerate(dataset[col]):
                if row > 10.0:
                    dataset[col][i] = dataset[col][i] / 1000
        return dataset

    def filter_null(self, dataset): #filter null or empty values
        nan_counts = dataset.isna().sum()
        print(nan_counts)
        for column in dataset.columns:
            if dataset[column].dtype != np.float64:
                continue
            nan_count = dataset[column].isna().sum()
            if nan_count > 0:
                median_value = dataset[column].median()
                dataset[column].fillna(median_value, inplace=True)
        return dataset
    
    def clean_dataset(self, dataset):
        print('Filtering dataset')
        dataset = self.convert_to_float(dataset)
        print('Converting to float')
        dataset = self.filter_dataset(dataset)
        print('Filtering null')
        dataset = self.filter_null(dataset)
        return dataset
    
    def prepare_training_data(self, dataset, feature_column, target_column, test_size=0.3, random_state=42):
        X = dataset[feature_column]
        y = dataset[target_column]

        # class_mapping = {1: 0,2: 1,3: 2}
        # y = [class_mapping[label] for label in y]
        y = pd.DataFrame(y, columns=['Level'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        y_train_count = y_train.value_counts()
        y_test_count = y_test.value_counts()

        y_train = y_train.to_numpy().reshape(-1)
        y_test = y_test.to_numpy().reshape(-1)

        print('y_train_count')
        print(y_train_count)

        print('y_test_count')
        print(y_test_count)

        return X_train, X_test, y_train, y_test
    
    def calculate_confusion_matrix(self, y_test, y_pred, ax, label=''):
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        sns.heatmap(conf_matrix_percent, ax=ax, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(3), yticklabels=range(3))
        ax.set_xlabel('Predicted Labels')
        ax.set_ylabel('True Labels')
        ax.set_title('Confusion Matrix - ' + label)
    def test(self):
        print('Test Class')
    def normalizer(self, X_train, X_test):
        scaler = MinMaxScaler()
        x_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        return x_train, X_test

    def plot_roc_curve_m(self, model, X_test, y_test, ax):
        # plot roc curve for multiclass classification
        y_pred = model.predict_proba(X_test)
        n_classes = y_pred.shape[1]
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test, y_pred[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        lw = 2
        ax.plot(fpr["micro"], tpr["micro"],
                label='micro-average ROC curve (area = {0:0.2f})'
                    ''.format(roc_auc["micro"]),
                color='deeppink', linestyle=':', linewidth=4)
        
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], lw=lw,
                    label='ROC curve of class {0} (area = {1:0.2f})'
                    ''.format(i, roc_auc[i]))
            
        ax.plot([0, 1], [0, 1], 'k--', lw=lw)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend()
        return ax
    
    def draw_roc_curve(self, X_train, X_test, y_train, y_test,save_location='', svmp = '', knnp ='', treep='', rfp='', random_state=42):
        if svmp == '':
            clf_svm = svm.SVC(random_state=random_state)
        else:
            clf_svm = svm.SVC(**svmp, random_state=random_state, probability=True)
        clf_svm.fit(X_train, y_train)

        if knnp == '':
            clf_knn = neighbors.KNeighborsClassifier()
        else:
            clf_knn = neighbors.KNeighborsClassifier(**knnp)
        clf_knn.fit(X_train, y_train)

        if treep == '':
            clf_tree = tree.DecisionTreeClassifier(random_state=random_state)
        else:
            clf_tree = tree.DecisionTreeClassifier(**treep, random_state=random_state)
        clf_tree.fit(X_train, y_train)

        if rfp == '':
            clf_rf = RandomForestClassifier(random_state=random_state)
        else:
            clf_rf = RandomForestClassifier(**rfp ,random_state=random_state)
        clf_rf.fit(X_train, y_train)
        models = {
            'SVM': clf_svm,
            'KNN': clf_knn,
            'Tree': clf_tree,
            'RF': clf_rf
        }
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
        # draw AUC ROC curve for each model in a 2x2 grid
        k = 0
        j = 0
        for i, (name, model) in enumerate(models.items()):
            roc = self.plot_roc_curve_m(model, X_test, y_test, ax=axes[k][j])
            axes[k][j].set_title(name)
            j += 1
            if j == 2:
                k += 1
                j = 0

    def train_ml_model(self, X_train, X_test, y_train, y_test,save_location='', svmp = '', knnp ='', treep='', rfp='', random_state=42):
        # Train SVM
        # print('Training SVM : ' + str(svmp))
        if svmp == '':
            clf_svm = svm.SVC(random_state=random_state)
        else:
            clf_svm = svm.SVC(**svmp, random_state=random_state)

        clf_svm.fit(X_train, y_train)
        y_pred_svm = clf_svm.predict(X_test)
        accuracy_svm = accuracy_score(y_test, y_pred_svm)

                # train svm with k-fold cross validation
        scores = cross_val_score(clf_svm, X_test, y_test, cv=5)
        print(scores)
        print("SVM Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("SVM accuracy:", accuracy_svm)

        # Train KNN
        # print('Training KNN : ' + str(knnp))
        if knnp == '':
            clf_knn = neighbors.KNeighborsClassifier()
        else:
            clf_knn = neighbors.KNeighborsClassifier(**knnp)
        clf_knn.fit(X_train, y_train)
        y_pred_knn = clf_knn.predict(X_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)

        scores = cross_val_score(clf_knn, X_test, y_test, cv=5)
        print(scores)
        print("KNN Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("KNN accuracy:", accuracy_knn)

        # Train Decision Tree
        # print('Training Decision Tree : ' + str(tree))
        if treep == '':
            clf_tree = tree.DecisionTreeClassifier(random_state=random_state)
        else:
            clf_tree = tree.DecisionTreeClassifier(**treep, random_state=random_state)
        clf_tree.fit(X_train, y_train)
        y_pred_tree = clf_tree.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_tree)

        scores = cross_val_score(clf_tree, X_test, y_test, cv=5)
        print(scores)
        print("Tree Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("Tree accuracy:", accuracy_tree)

        # Train Random Forest
        # print('Training Random Forest : ' + str(rfp))
        if rfp == '':
            clf_rf = RandomForestClassifier(random_state=random_state)
        else:
            clf_rf = RandomForestClassifier(**rfp ,random_state=random_state)
        clf_rf.fit(X_train, y_train)
        y_pred_rf = clf_rf.predict(X_test)
        accuracy_tree = accuracy_score(y_test, y_pred_rf)

        scores = cross_val_score(clf_rf, X_test, y_test, cv=5)
        print(scores)
        print("RF Cross Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
        print("RF accuracy:", accuracy_tree)

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(7, 7))
        models = {
            'SVM': clf_svm,
            'KNN': clf_knn,
            'Tree': clf_tree,
            'RF': clf_rf
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        k = 0
        j = 0

        for i, (name, model) in enumerate(models.items()):
            all_predictions = []
            all_true_labels = []
            for _,test_index in skf.split(X_test, y_test):
                XX_test = X_test.values[test_index]
                yy_test = y_test[test_index]
                all_predictions.extend(model.predict(XX_test))
                all_true_labels.extend(yy_test)

            self.calculate_confusion_matrix(all_true_labels, all_predictions, axes[k][j], label=name)
            j += 1
            if j == 2:
                k += 1
                j = 0

            # calculate sensitivity, specificity, precision, recall, f1 score
            print(name)
            print(classification_report(all_true_labels, all_predictions))
        plt.tight_layout()
        plt.show()
        if save_location != '':
            fig.savefig(save_location, dpi=600, bbox_inches='tight')
        

        

        return clf_svm, clf_knn, clf_tree, clf_rf
    
    def do_SMOTE(self, k_neighbors, X_train, y_train):
        sm = SMOTE(sampling_strategy='auto',k_neighbors=k_neighbors,  random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        return X_train, y_train
    
    def train_transformer(self, X_resampled_train, y_resampled_train, X_resampled_test, y_resampled_test):
        keras.utils.set_random_seed(812)
        tf.config.experimental.enable_op_determinism()

        y_train_encoded = to_categorical(y_resampled_train, num_classes=3)
        model = self.transformer_model(input_shape=(X_resampled_train.shape[1], 1))
        model.compile(optimizer=Adam(learning_rate=1e-3),
                loss=BinaryCrossentropy(),
                metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
        model.fit(X_resampled_train, y_train_encoded, validation_split=0.2, epochs=100, batch_size=4, callbacks=[early_stopping], verbose=0)


        y_pred = model.predict(X_resampled_test)
        y_pred_binary = [int(np.argmax(p)) for p in y_pred]
        conf_matrix = confusion_matrix(y_resampled_test, y_pred_binary)
        conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        # print(conf_matrix_percent)
        print(accuracy_score(y_resampled_test, y_pred_binary))

        plt.figure(figsize=(5, 5))
        sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=range(3), yticklabels=range(3))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()


    def transformer_model(self, input_shape):
        inputs = Input(shape=input_shape)
        attention_output = MultiHeadAttention(num_heads=1, key_dim=256)(inputs, inputs)
        x = GlobalAveragePooling1D()(attention_output)
        x = Dense(64, activation='relu', )(x)
        # x = Dropout(0.1)(x)
        x = Dense(128, activation='relu')(x)
        # x = Dropout(0.2)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.1)(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(1024, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        outputs = Dense(3, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model

        
    def dataset_analysis(self, color_dict, dataset, keys, save_location):
        # Define colors for each class
        # color_dict = {0: 'green', 1: 'blue', 2: 'red'}
        # Get the unique keys excluding 'Level'
        keys_without_level = [key for key in keys if key != 'Level']

        # Calculate the number of rows and columns for subplots
        num_rows = 6 #(len(keys_without_level)) // 3  # Add 1 to round up
        num_cols = 4
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 24))

        for idx, key in enumerate(keys_without_level):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

        # for key in keys:
            if key == 'Level':
                continue

            plt.figure()  # Create a new figure for each key
            for class_label, color in color_dict.items():
                # print(class_label)
                data = dataset[dataset['Level'] == class_label][key]
                if not data.empty:
                    data.plot(kind='hist', alpha=0.4, color=color, bins=50, label=class_label, ax=ax)

            ax.set_title(key)
            ax.legend()

        if len(keys_without_level) % 2 != 0:
            fig.delaxes(axes[-1, -1])

        plt.tight_layout()
        ax.figure.savefig(save_location, dpi=600, bbox_inches='tight')
        plt.close()
    def draw_correration_matrix(self, dataset, save_location):
        correlation =dataset.copy()
        corr= correlation.corr(method='spearman')
        fig, ax = plt.subplots(figsize=(16,16))
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(data=corr, annot=True,ax=ax,mask=mask)

        plt.title('Correlation Matrix')
        plt.savefig('result/dataset_analysis/correlation_matrix.png', dpi=600, bbox_inches='tight')
        plt.close()

    def feature_distance(self, data, label = '', save_location = ''):
        corr = data.corr(method = 'spearman')
        dist_linkage = linkage(squareform(1 - abs(corr)), 'complete')
        
        plt.figure(figsize = (10, 8), dpi = 300)
        dendro = dendrogram(dist_linkage, labels=data.columns, leaf_rotation=90)
        # plt.title(f'Feature Distance in {label} Dataset', weight = 'bold', size = 20)
        plt.savefig(save_location, dpi=600, bbox_inches='tight')
        plt.close()

    def parameter_tuning(self,X_train, y_train, 
                         param_grid_rf,svm_param_grid, knn_param_gird, df_param_grid):
        rf = RandomForestClassifier(random_state = 42)
        fbeta_scorer = make_scorer(fbeta_score, beta=2, average='micro')  # change beta as needed
        grid_search_f1 = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=5, scoring= fbeta_scorer)
        grid_search_f1.fit(X_train, y_train) 
        cv_results = grid_search_f1.cv_results_
        best_params_rf = grid_search_f1.best_params_
        print(best_params_rf)

        svm_model = svm.SVC()
        fbeta_scorer = make_scorer(fbeta_score, beta=2, average='micro')  # change beta as needed
        grid_search_f1 = GridSearchCV(estimator=svm_model, param_grid=svm_param_grid, cv=5, n_jobs=5, scoring= fbeta_scorer)
        grid_search_f1.fit(X_train, y_train)
        cv_results = grid_search_f1.cv_results_
        best_params_svm = grid_search_f1.best_params_
        print(best_params_svm)

        knn_model = neighbors.KNeighborsClassifier()
        fbeta_scorer = make_scorer(fbeta_score, beta=2, average='micro')  # change beta as needed
        grid_search_f1 = GridSearchCV(estimator=knn_model, param_grid=knn_param_gird, cv=5, n_jobs=5, scoring= fbeta_scorer)
        grid_search_f1.fit(X_train, y_train)
        cv_results = grid_search_f1.cv_results_
        best_params_knn = grid_search_f1.best_params_
        print(best_params_knn)

        decisiontree_model = tree.DecisionTreeClassifier()
        fbeta_scorer = make_scorer(fbeta_score, beta=2, average='micro')  # change beta as needed
        grid_search_f1 = GridSearchCV(estimator=decisiontree_model, param_grid=df_param_grid, cv=5, n_jobs=5, scoring= fbeta_scorer)
        grid_search_f1.fit(X_train, y_train)
        cv_results = grid_search_f1.cv_results_
        best_params_dt = grid_search_f1.best_params_
        print(best_params_dt)

        return best_params_rf, best_params_svm, best_params_knn, best_params_dt
    
    def svm_classification_plot(self, X_train, X_test,y_train, y_test, best_params_svm, save_location = ''):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        svm_model = svm.SVC(**best_params_svm, random_state=42)
        svm_model.fit(X_train_pca, y_train)
        # Create a meshgrid to plot decision boundary
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Predict class for each point in meshgrid
        Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # plot tsne
        # plt.figure(figsize=(10, 8))
        # tsne = TSNE(n_components=2, random_state=42)
        # X_train_tsne = tsne.fit_transform(X_train_pca)
        # X_test_tsne = tsne.fit_transform(X_test_pca)
        
        # df_subset =  pd.DataFrame()
        # df_subset['y'] = y_train
        # df_subset['tsne-2d-one'] = X_train_tsne[:,0]
        # df_subset['tsne-2d-two'] = X_train_tsne[:,1]

        # plt.figure(figsize=(16,10))
        # sns.scatterplot(
        #     x="tsne-2d-one", y="tsne-2d-two",
        #     hue="y",
        #     palette=sns.color_palette("hls", 10),
        #     data=df_subset,
        #     legend="full",
        #     alpha=0.3
        # )


        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.4)
        # 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG'
        # print decision boundary parameter for explainability
        # Plot data points
        plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, marker='o', cmap='BrBG', label='Training Data')
        plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, marker='s', cmap='Accent', label='Testing Data')

        # Add labels and legend
        plt.xlabel('PCA 0')
        plt.ylabel('PCA 1')
        plt.title('SVM Classification Plot')
        plt.legend()
        plt.savefig(save_location, dpi=600, bbox_inches='tight')
        # plt.show()
        return svm_model

    def lime_explain(self, features, X_train,X_test, model, index, save_location = ''):

        # shorten feature names
        t_features = features.copy()
        
        # # capitalize the first letter of each word
        # for i in range(len(t_features)):
        #     t_features[i] = t_features[i].title()
            

        # # only keep the capital characters in a word
        # for i in range(len(t_features)):
        #     word = t_features[i]
        #     t_features[i] = ''.join([c for c in word if c.isupper()])

        t_features = ['AP', 'AU', 'DA', 'OH', 'GR', 'CLD', 'BD', 'O', 'SM', 'PS', 'CP', 'COB', 'F', 'WL', 'SOB', 'W', 'SD', 'COFN', 'FC', 'DC', 'SN']
        print(features)
        print(t_features)
        explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=t_features, class_names=['Low', 'Medium', 'High'], discretize_continuous=True)
        exp = explainer.explain_instance(X_test.values[index], model.predict_proba, num_features=21, top_labels=1)

        if save_location != '':
            exp.save_to_file(save_location)
        else:
            exp.show_in_notebook(show_table=True, show_all=True)

    def plot_decision_tree(self, model, features, save_location = '', fontsize=16, dimension=(60, 30)):
        plt.figure(figsize=dimension)
        plot_tree(model, filled=True, feature_names=features, class_names=['Low', 'Medium', 'High'], fontsize=fontsize)
        if save_location != '':
            plt.savefig(save_location, dpi=600, bbox_inches='tight')
        # plt.show()