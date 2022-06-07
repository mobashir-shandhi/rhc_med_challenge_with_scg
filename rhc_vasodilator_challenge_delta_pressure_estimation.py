# -*- coding: utf-8 -*-
"""
This code will estimate changes in the pulmonary artery pressure or pulmonary capillary wedge pressure due to 
vasodilator challenge during Right Heart  Catheterization procedure using simulataneously recorded seismocardiogram
signals recorded using a wearable chest patch.

@author: mobashir shandhi
"""

import numpy as np 
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from dtw import dtw
from sklearn import mixture
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR, LinearSVR


# Next line to silence pyflakes. This import is needed.
Axes3D




################function to get DTW distance with outlier removal #################
def get_dtw_distance(X_sub,bl_med_sub):       
    outlier_percentage=20   # defining outlier percentage to remove from a distribution
    n_components=3          # number of components to extract from PCA of the frames for outlier
    #n_neighbors=10
    X_sub_bl=X_sub[bl_med_sub==0]  # get baseline frames
    X_sub_med=X_sub[bl_med_sub==1]  # get vasodilator challenge frames
    
    

    
    ### removing artifacts/outlier beats using low dimensional structure of SCG frames to remove outlier with GMM
    # defining manifold/dimension reduction technique used
    manifold_method=PCA(n_components=n_components)  
    # defining gaussina mixture model used to get the distribution
    gmm = mixture.GaussianMixture(n_components=1, n_init=10, covariance_type='full')

    # outlier removal for baseline SCG frames 
    Y_bl = manifold_method.fit_transform(X_sub_bl)
    gmm.fit(Y_bl)
    densities_bl = gmm.score_samples(Y_bl)
    density_threshold = np.percentile(densities_bl, outlier_percentage)
    #anomalies_bl = Y_bl[densities_bl < density_threshold]    
    X_sub_bl=X_sub_bl[densities_bl > density_threshold] # keepking the non-outlier frames

    # outlier removal for vasodilator challenge SCG frames 
    Y_med = manifold_method.fit_transform(X_sub_med)
    gmm.fit(Y_med)
    densities_med = gmm.score_samples(Y_med)
    density_threshold = np.percentile(densities_med, outlier_percentage)
    #anomalies_med = Y_med[densities_med < density_threshold]    
    X_sub_med=X_sub_med[densities_med > density_threshold]  # keepking the non-outlier frames
    
        
    #get the dtw distance between the ensemble-averaged frames: one averaged frame for baseline and one averaged frame for vasodilator challenge
    alignment = dtw(np.mean(X_sub_med,axis=0), np.mean(X_sub_bl,axis=0), step_pattern="asymmetric", keep_internals=True)

    return alignment.distance



# function to create bland_altman_plot
def bland_altman_plot(data1, data2, *args, **kwargs):
    data1     = np.asarray(data1)
    data2     = np.asarray(data2)
    mean      = np.mean([data1, data2], axis=0)
    diff      = data1 - data2                   # Difference between data1 and data2
    md        = np.mean(diff)                   # Mean of the difference
    sd        = np.std(diff, axis=0)            # Standard deviation of the difference

    plt.scatter(mean, diff, *args, **kwargs)
    plt.axhline(md,           color='gray', linestyle='--')
    plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96*sd, color='gray', linestyle='--')
    
# function to calculate R-square
def calculate_r2(x,y):
    correlation_matrix = np.corrcoef(x,y)
    correlation_xy = correlation_matrix[0,1]
    r_squared = correlation_xy**2     
    return r_squared


#close all pre-existing plots and modify plot settings
plt.close('all')
plt.interactive(False)






##### Import Data: pressure values in csv and corresponing SCG frames in mat file
#### 20s PA frame during baseline and vasodilator challenge from 20 subjects
df = pd.read_csv('RHC_PA_med_features_20s_sys_dias_20201203.csv')
scg_frame = loadmat('RHC_PA_med_frames_20s_sys_dias_20201203.mat')


#### 20s PCWP frame during baseline and vasodilator challenge from 19 subjects
#df = pd.read_csv('RHC_PCWP_med_features_20s_sys_dias_20201203.csv')
#scg_frame = loadmat('RHC_PCWP_med_frames_20s_sys_dias_20201203.mat')



###### get corresponding SCG frames; systolic and diastolic frames separatele
ax_sys_frame=scg_frame['a_x_systole_frame_all_sub']
ay_sys_frame=scg_frame['a_y_systole_frame_all_sub']
az_sys_frame=scg_frame['a_z_systole_frame_all_sub']
at_sys_frame=np.sqrt(ax_sys_frame**2+ay_sys_frame**2+az_sys_frame**2)  # calculating the magnitude axis
ax_dias_frame=scg_frame['a_x_diastole_frame_all_sub']
ay_dias_frame=scg_frame['a_y_diastole_frame_all_sub']
az_dias_frame=scg_frame['a_z_diastole_frame_all_sub']
at_dias_frame=np.sqrt(ax_dias_frame**2+ay_dias_frame**2+az_dias_frame**2) # calculating the magnitude axis

#get the list of frames to identify which frame belongs to baseline and vasodilator respectively
bl_med_list=scg_frame['bl_med_list_all_sub']  

#get the list of subjects to identify which frame belongs to which subject
groups=scg_frame['subject_ids_for_frames']

#get HF classification (HFrEF and HFpEF)
hf_class=df['HF Classification']

# get target variables and corresponding subject ids
delta_pamp=df['Delta PAP'].values
#delta_pcwp=df['Delta PCWP'].values
subject_ids=df['Subject_ID'].values




#### declare target variable: Delta PAM or Delta PCWP
y=delta_pamp
#y=delta_pcwp
groups=groups.flatten()
bl_med_list=bl_med_list.flatten()


# declaring systolic and diastolic time intervals to extract corresponding sub-frames from SCG frames
# For systolic phase
ao_start=25   # Isovolumetric Contration (IVC) Start
ao_end=150    # IVC End
ac_start=200    # Left ventricular (LV) Ejection Start
ac_end=500      # LV Ejection End


# For Diastolic phase
dias1_start=300 # atrial systole/kick start
dias1_end=500   # atrial systole/kick end
dias2_start=0   # rapid ejection start
dias2_end=300   # rapid ejection end





############Get DTW for full signal, DTW for AO, DTW for AC  
# Definding number of sub_frames to calculate DTW distances
# 4 sub-frames (2 for systolic, 2 for diastolic) + 1 sub-frame for the whole systolic + 1 sub-frame for the whole diastolic phase for each SCG axis: AX, AY, AZ, AT   
# in total 6x4=24 sub-frames  
number_of_SCG_sub_frames=24 

#creating empty matrix to hold the DTW distances
dtw_distance_matrix=np.zeros(shape=(len(np.unique(groups)),number_of_SCG_sub_frames))    
j=0
for subjectNo in np.unique(groups):
    bl_med_sub=bl_med_list[groups==subjectNo]
    
    #########from systolic frames#################
    dtw_distance_matrix[j,0]=get_dtw_distance(ax_sys_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,1]=get_dtw_distance(ay_sys_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,2]=get_dtw_distance(az_sys_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,3]=get_dtw_distance(at_sys_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,4]=get_dtw_distance(ax_sys_frame[groups==subjectNo,ao_start:ao_end],bl_med_sub)
    dtw_distance_matrix[j,5]=get_dtw_distance(ay_sys_frame[groups==subjectNo,ao_start:ao_end],bl_med_sub)
    dtw_distance_matrix[j,6]=get_dtw_distance(az_sys_frame[groups==subjectNo,ao_start:ao_end],bl_med_sub)
    dtw_distance_matrix[j,7]=get_dtw_distance(at_sys_frame[groups==subjectNo,ao_start:ao_end],bl_med_sub)
    dtw_distance_matrix[j,8]=get_dtw_distance(ax_sys_frame[groups==subjectNo,ac_start:ac_end],bl_med_sub)
    dtw_distance_matrix[j,9]=get_dtw_distance(ay_sys_frame[groups==subjectNo,ac_start:ac_end],bl_med_sub)
    dtw_distance_matrix[j,10]=get_dtw_distance(az_sys_frame[groups==subjectNo,ac_start:ac_end],bl_med_sub)
    dtw_distance_matrix[j,11]=get_dtw_distance(at_sys_frame[groups==subjectNo,ac_start:ac_end],bl_med_sub)

    #########from diastolic frames#################
    dtw_distance_matrix[j,12]=get_dtw_distance(ax_dias_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,13]=get_dtw_distance(ay_dias_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,14]=get_dtw_distance(az_dias_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,15]=get_dtw_distance(at_dias_frame[groups==subjectNo],bl_med_sub)
    dtw_distance_matrix[j,16]=get_dtw_distance(ax_dias_frame[groups==subjectNo,dias1_start:dias1_end],bl_med_sub)
    dtw_distance_matrix[j,17]=get_dtw_distance(ay_dias_frame[groups==subjectNo,dias1_start:dias1_end],bl_med_sub)
    dtw_distance_matrix[j,18]=get_dtw_distance(az_dias_frame[groups==subjectNo,dias1_start:dias1_end],bl_med_sub)
    dtw_distance_matrix[j,19]=get_dtw_distance(at_dias_frame[groups==subjectNo,dias1_start:dias1_end],bl_med_sub)
    dtw_distance_matrix[j,20]=get_dtw_distance(ax_dias_frame[groups==subjectNo,dias2_start:dias2_end],bl_med_sub)
    dtw_distance_matrix[j,21]=get_dtw_distance(ay_dias_frame[groups==subjectNo,dias2_start:dias2_end],bl_med_sub)
    dtw_distance_matrix[j,22]=get_dtw_distance(az_dias_frame[groups==subjectNo,dias2_start:dias2_end],bl_med_sub)
    dtw_distance_matrix[j,23]=get_dtw_distance(at_dias_frame[groups==subjectNo,dias2_start:dias2_end],bl_med_sub)
    
    j=j+1




#############Create a training-testing and a seperate validation set using random sampling#############
groups=np.unique(groups)
X_train_test, X_val, y_train_test, y_val, groups_train_test, groups_val, hf_class_train_test, hf_class_val= train_test_split(dtw_distance_matrix, y, groups,hf_class, test_size=0.25, random_state=42, stratify=hf_class)

#scaling the data
standard_scaler=StandardScaler().fit(X_train_test)
X_train_test = standard_scaler.transform(X_train_test)
X_val = standard_scaler.transform(X_val)


############Select 5 features using sequential feature selection using training-testing set data ###########
#define a regression model for the feature selection
reg_model = LinearSVR()

#define feature names from the input DTW array
feature_set=['AX_500ms', 'AY_500ms', 'AZ_500ms', 'AT_500ms','AX_IVC','AY_IVC','AZ_IVC','AT_IVC','AX_EJ','AY_EJ','AZ_EJ','AT_EJ',
             'AX_Dias_500ms','AY_Dias_500ms','AZ_Dias_500ms','AT_Dias_500ms','AX_Dias_active','AY_Dias_active','AZ_Dias_active','AT_Dias_active','AX_Dias_passive','AY_Dias_passive','AZ_Dias_passive','AT_Dias_passive']


sfs1 = SequentialFeatureSelector(reg_model, n_features_to_select=5, direction='forward',scoring='neg_root_mean_squared_error')
sfs1.fit(X_train_test, y_train_test)
select_features=sfs1.get_feature_names_out(input_features=feature_set)
#print selected features
print ('Selected features: ')
print  (select_features)
print ('\n')


#transform the data to only use the selected features for the model development
X_select_train_test = sfs1.transform(X_train_test)
X_select_val = sfs1.transform(X_val)

X_train_test=X_select_train_test
X_val=X_select_val


####get feature importance with SVR##########
fit=reg_model.fit(X_train_test,y_train_test)
weights=fit.coef_
print('Feature Weights: ')
print(weights)
print('\n')


##############perform grid search on training-testing set############
estimator=svm.SVR() #regression model for this analysis
param_dist = {
            'kernel':('linear', 'rbf', 'poly','sigmoid'), 
            'C':[ 0.1, 0.5, 1, 3, 5, 10, 100, 1000],
            'degree' : [3,8],
            'coef0' : [0.0001, 0.001, 0.01,0.1,0.5, 1.0, 10],
            'gamma' : ('auto','scale')
            }

num_subjects=len(np.unique(groups_train_test))
cv=GroupKFold(n_splits=num_subjects)


grid_search = GridSearchCV(estimator=estimator, param_grid=param_dist, cv = cv,  n_jobs = -1,verbose=True,scoring='neg_root_mean_squared_error')
grid_search.fit(X_train_test,y_train_test, groups=groups_train_test)

best_grid = grid_search.best_estimator_
print ("Best Grid")
print (best_grid)
#choosing the reg model from grid search
reg_model=grid_search.best_estimator_ 



###################### Regression ##############
############# perform leave-one-out-cross-validation on the training-testing set #############
logo = LeaveOneGroupOut()

#initialize vectore to keep score for each fold
y_train_test_predictions = np.zeros(y_train_test.shape)



for train, test in logo.split(X_train_test, y_train_test, groups=groups_train_test):

    #perform train test split for the current fold
    X_train, X_test, y_train, y_test = X_train_test[train], X_train_test[test], y_train_test[train], y_train_test[test]   
   
    model=reg_model # regression model from the grid search portion

    model_trained = model.fit(X_train , y_train)
    y_predicted = model_trained.predict(X_test)  
    

    y_train_test_predictions[test] = y_predicted

    # printing the number in output console
    print('Subject No= ' + str(np.mean(groups[test])))
    print('True Delta PAMP = ' + str(np.mean(y_test))+' Predicted Delta PAMP = ' + str(np.mean(y_predicted)))


# calculate RMSE, NRMSE, R-square and pearson p_value between actual and predicted value.
rmse=np.sqrt(mean_squared_error(y_train_test, y_train_test_predictions))
rmse_normalized=rmse/np.mean(y_train_test)
r_squared=calculate_r2(y_train_test, y_train_test_predictions)
p_value=sp.stats.pearsonr(y_train_test, y_train_test_predictions)


#print results
print('Training-Testing Set Results')
print('Training-Testing RMSE= ' + str(rmse))
print('Rsquared= ' + str(r_squared))
print('Pearson Corr-Coeff and P-value= ' + str(p_value))
print('Normalized RMSE = ' + str(rmse_normalized))


#plot out of fold predictions against actual PAMP
fig = plt.figure();
cm=plt.cm.get_cmap('jet')
pl=plt.scatter(y_train_test, y_train_test_predictions , c= groups_train_test, cmap= cm)
plt.title('Train-Test Set OOF R^2= ' + str(r_squared) +' \nNRMSE= ' + str(rmse_normalized))
plt.colorbar(pl)
plt.show()


# Bland Altman Plot for the y actual and y predicted
fig = plt.figure();
bland_altman_plot(y_train_test, y_train_test_predictions)
plt.title('Bland-Altman Plot')
plt.ylabel('Actual-Predicted')
plt.xlabel('(Actual+Predicted)/2')
plt.show()


############# validate the model on the independent validation set #############
y_val_predictions = np.zeros(y_val.shape)

#get predicted pressure change for each subject in the validation set
for subjectNo in np.unique(groups_val):

    #Train on the whole train set and test on the validation set subjects
    X_val_sub = X_val[groups_val==subjectNo]
    y_val_sub = y_val[groups_val==subjectNo]
    
    
    #model training using traing_testing data
    model_trained = model.fit(X_train_test ,y_train_test)
    y_predicted = model_trained.predict(X_val_sub)  
    
    y_val_predictions[groups_val==subjectNo] = y_predicted
    
    print('Subject No= ' + str(np.mean(groups_val[groups_val==subjectNo])))
    print('True Delta PAMP = ' + str(np.mean(y_test))+' Predicted Delta PAMP = ' + str(np.mean(y_predicted)))
    
    
rmse_val=np.sqrt(mean_squared_error(y_val, y_val_predictions))
rmse_val_normalized=rmse_val/np.mean(y_val)
r_squared_val=calculate_r2(y_val, y_val_predictions)
p_value_val=sp.stats.pearsonr(y_val, y_val_predictions)


#print results
print('Validation Set Results')
print('RMSE= ' + str(rmse_val))
print('Rsquared= ' + str(r_squared_val))
print('Pearson Corr-Coeff and P-value= ' + str(p_value_val))
print('Normalized RMSE = ' + str(rmse_val_normalized))


#plot out of fold predictions against actual PEP
fig = plt.figure();
cm=plt.cm.get_cmap('jet')
pl=plt.scatter(y_val , y_val_predictions , c= groups_val, cmap= cm)
plt.title('Validation Set OOF R^2= ' + str(r_squared_val) +' \nRMSE Normalized= ' + str(rmse_val_normalized))
plt.colorbar(pl)
plt.show()


# Bland Altman Plot for the y actual and y predicted
fig = plt.figure();
bland_altman_plot(y_val, y_val_predictions)
plt.title('Bland-Altman Plot for Validation Set')
plt.ylabel('Actual-Predicted')
plt.xlabel('(Actual+Predicted)/2')
plt.show()
