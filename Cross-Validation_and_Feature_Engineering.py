#!/usr/bin/env python
# coding: utf-8

# **Run the following two cells before you begin.**

# In[1]:


get_ipython().run_line_magic('autosave', '10')


# In[109]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

get_ipython().run_line_magic('matplotlib', 'inline')


# **First, import the cleaned data set. Then, select the features from the DataFrame of the case study data.**
#     
# These features should be: `'LIMIT_BAL'`, `'EDUCATION'`, `'MARRIAGE'`, `'AGE'`, `'PAY_1'`, `'BILL_AMT1'`, `'BILL_AMT2'`, `'BILL_AMT3'`, `'BILL_AMT4'`, `'BILL_AMT5'`, `'BILL_AMT6'`, `'PAY_AMT1'`, `'PAY_AMT2'`, `'PAY_AMT3'`, `'PAY_AMT4'`, `'PAY_AMT5'`, AND `'PAY_AMT6'`.

# In[110]:


import pandas as pd
import matplotlib as mpl
import seaborn as sns
df = pd.read_csv('cleaned_data.csv')
features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
corr = df[features_response].corr()
mpl.rcParams['figure.dpi'] = 400 #high res figures
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
            center=0)


# In[111]:


# Import data set
df = pd.read_csv('cleaned_data.csv')

features =pd.DataFrame(df,columns =['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5','PAY_AMT6'])


# In[112]:


# Create features list
features =['LIMIT_BAL', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_1', 'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']


# In[113]:


# Create features list
features = features[:-1]


# In[114]:


X= df[features].values


# _____________________________________________________
# **Next, make a 80:20 train/test split using a random seed of 24.**

# In[115]:



X_train,X_test,y_train,y_test=train_test_split(X,df['default payment next month'].values,test_size=0.2,random_state=24)


# In[116]:


from sklearn.preprocessing import MinMaxScaler
min_max_sc = MinMaxScaler()


# _____________________________________________________
# **Next, instantiate a logistic regression model with the `saga` solver, L1 penalty, and set `max_iter` to 1,000 as we want the solver to have enough iterations to find a good solution.**

# In[44]:


lr = LogisticRegression(solver = 'saga',penalty ='l1',max_iter= 1000)


# _____________________________________________________
# **Next, import the `Pipeline` class and create a `Pipeline` with the scaler and the logistic regression model, using the names `'scaler'` and `'model'` for the steps, respectively.**

# In[117]:


from sklearn.pipeline import Pipeline
scale_lr_pipeline = Pipeline(steps =[('scaler',min_max_sc),('model',lr)])


# _____________________________________________________
# **Now, use the `get_params` method to view the parameters from each stage of the pipeline.**

# In[50]:


# Use `get_params`
scale_lr_pipeline.get_params()


# **Use the `set_params` method to change the the `model__C` parameter to 2.**

# In[53]:


# View what `model__C` is set to currently
scale_lr_pipeline.get_params()['model__C']


# In[52]:


# Change `model__C` to 2
scale_lr_pipeline.set_params(model__C = 2 )


# _____________________________________________________
# **Then, create a smaller range of C values to test with cross-validation, as these models will take longer to train and test with more data than our previous activities.**
# 
# **Use C_vals = [$10^2$, $10$, $1$, $10^{-1}$, $10^{-2}$, $10^{-3}$].**
# 
# 
# <details>
#     <summary>Hint:</summary>
#     Recall that exponents in Python use the ** operator.
# </details>

# In[ ]:





# Now, define `k_folds` using `StratifiedKFold`. The number of folds should be 4. Set the random state to 1.

# In[59]:


from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


# In[60]:


def plot_kfolds(k_folds_iterator):
    fold_counter = 0
    for train_index, test_index in k_folds_iterator.split(X_syn_train, y_syn_train):
 
        #Axis to hold the plot of this fold
        ax = plt.subplot(n_folds,1,fold_counter+1)
 
        #Background rectangle representing all samples
        n_train_samples = len(y_syn_train)
        rect = mpl.patches.Rectangle(xy=(0,0), width=n_train_samples, height=1)
        ax.add_patch(rect)
 
        #Plot each testing sample from this fold as a vertical line
        for this_text_ix in test_index:
            ax.plot([this_text_ix, this_text_ix], [0, 1], color='orange',
                    linewidth=0.75)
 
        #Plot formatting
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, n_train_samples)
        ax.set_ylim(0,1)
 
        #Subplot titles
        if fold_counter == 0:
            ax.text(0.26, 1.2, 'Training data,',
                    transform=ax.transAxes, backgroundcolor = 'blue')
            ax.text(0.45, 1.2, 'testing data:',
                    transform=ax.transAxes, backgroundcolor = 'orange')
            ax.text(0.62, 1.2, 'fold {}'.format(fold_counter+1), transform=ax.transAxes)
        else:
            ax.text(0.45, 1.2, 'Fold {}'.format(fold_counter+1), transform=ax.transAxes)
 
        fold_counter += 1
 
    plt.tight_layout()


# In[118]:


n_folds = 4
k_folds = KFold(n_splits=n_folds, shuffle=False, random_state=1)


# In[119]:


plot_kfolds(k_folds)


# _____________________________________________________
# **Next, make a new version of the `cross_val_C_search` function, called `cross_val_C_search_pipe`. Instead of the model argument, this function will take a pipeline argument. The changes inside the function will be to set the `C` value using `set_params(model__C = <value you want to test>)` on the pipeline, replacing the model with the pipeline for the fit and `predict_proba` methods, and accessing the `C` value using `pipeline.get_params()['model__C']` for the printed status update.**

# In[120]:


from sklearn.metrics import roc_curve


# In[ ]:





# _____________________________________________________
# **Now, run this function as in the previous activity, but using the new range of `C` values, the pipeline you created, and the features and response variable from the training split of the case study data.**
# 
#     You may see warnings here, or in later steps, about the non-convergence of the solver; you could experiment with the `tol` or `max_iter`` options to try and achieve convergence, although the results you obtain with `max_iter = 1000` are likely to be sufficient.

# In[ ]:





# _____________________________________________________
# **Plot the average training and testing ROC AUC across folds, for each `np.log(C_vals)` value.**

# In[105]:


C_val_exponents = np.linspace(3,-3,13)
C_val_exponents


# In[106]:


C_vals = np.float(10)**C_val_exponents
C_vals


# In[107]:


from sklearn.metrics import roc_curve


# In[108]:


def cross_val_C_search(k_folds, C_vals, model, X, Y):


# _____________________________________________________
# **Up next, create interaction features for the case study data using scikit-learn's `PolynomialFeatures`. You should use 2 as the degree of polynomial features. Confirm that the number of new features makes sense.**

# In[66]:


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[67]:


import numpy as np
X_poly = np.linspace(-3,5,81)
print(X_poly[:5], '...', X_poly[-5:])


# In[68]:


X_synthetic, y_synthetic = make_classification(
    n_samples=1000, n_features=200,
    n_informative=3, n_redundant=10,
    n_repeated=0, n_classes=2,
    n_clusters_per_class=2, weights=None,
    flip_y=0.01, class_sep=0.8,
    hypercube=True, shift=0.0,
    scale=1.0, shuffle=True,
    random_state=24
)


# In[69]:


for plot_index in range(4):
     plt.subplot(2,2,plot_index+1)
     plt.hist(X_synthetic[:,plot_index])
     plt.title('Histogram for feature {}'.format(plot_index+1))
plt.tight_layout()


# _____________________________________________________
# **Finally, repeat the cross-validation procedure and observe the model performance now.**

# In[70]:


# Using the new features, make a 80:20 train/test split using a random seed of 24.**
X_syn_train, X_syn_test, y_syn_train, y_syn_test = train_test_split(
    X_synthetic, y_synthetic,
    test_size=0.2, random_state=24
)
lr_syn = LogisticRegression(solver='liblinear', penalty='l1', C=1000, random_state=1)
lr_syn.fit(X_syn_train, y_syn_train)


# In[71]:


lr_syn.fit(X_syn_train, y_syn_train)


# In[72]:


y_syn_train_predict_proba = lr_syn.predict_proba(X_syn_train)
roc_auc_score(y_syn_train, y_syn_train_predict_proba[:,1])


# In[73]:


y_syn_test_predict_proba = lr_syn.predict_proba(X_syn_test)
roc_auc_score(y_syn_test, y_syn_test_predict_proba[:,1])


# In[103]:


# Call the cross_val_C_search_pipe() function using the new training data.
# All other parameters should remain the same.
# Note that this training may take a few minutes due to the larger number of features.
from sklearn.pipeline import Pipeline
scale_lr_pipeline = Pipeline(steps=[('scaler', min_max_sc), ('model', lr_syn)])


# In[104]:


from sklearn.preprocessing import PolynomialFeatures
make_interactions = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)


# In[75]:


C_val_exponents = np.linspace(3,-3,13)
C_val_exponents


# In[76]:


C_vals = np.float(10)**C_val_exponents
C_vals


# In[96]:


def cross_val_C_search(k_folds, C_vals, model, X, Y):


# In[93]:


n_folds = k_folds.n_splits
cv_train_roc_auc = np.empty((n_folds, len(C_vals)))
cv_test_roc_auc = np.empty((n_folds, len(C_vals)))


# In[102]:


#Fit the model on the training data
model.fit(X_cv_train, y_cv_train)


# In[122]:


# Plot the average training and testing ROC AUC across folds, for each C value.
from sklearn.metrics import roc_curve
#Get the training ROC AUC
y_cv_train_predict_proba = model.predict_proba(X_cv_train)
cv_train_roc_auc[fold_counter, c_val_counter] = roc_auc_score(y_cv_train, y_cv_train_predict_proba[:,1])


# In[87]:


cv_test_roc = [[]]*len(C_vals)


# In[99]:


for c_val_counter in range(len(C_vals)):
    #Set the C value for the model object
    model.C = C_vals[c_val_counter]
    #Count folds for each value of C
    fold_counter = 0


# **Take a look at the above graph. Does the average cross-validation testing performance improve with the interaction features? Is regularization useful?**

# In[ ]:




