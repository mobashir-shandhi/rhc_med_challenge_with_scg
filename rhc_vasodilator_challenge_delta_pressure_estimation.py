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
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D
from dtw import dtw
from sklearn import mixture
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs



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
df = pd.read_csv('D:\Onedrive Gatech\OneDrive - Georgia Institute of Technology\Journal and Conferences\Journal on RHC Med Challenge\Data\Features\RHC_PA_med_features_20s_sys_dias_20201203.csv')
scg_frame = loadmat('D:\Onedrive Gatech\OneDrive - Georgia Institute of Technology\Journal and Conferences\Journal on RHC Med Challenge\Data\Features/RHC_PA_med_frames_20s_sys_dias_20201203.mat')


#### 20s PCWP frame during baseline and vasodilator challenge from 19 subjects
#df = pd.read_csv('D:\Onedrive Gatech\OneDrive - Georgia Institute of Technology\Journal and Conferences\Journal on RHC Med Challenge\Data\Features\RHC_PCWP_med_features_20s_sys_dias_20201203.csv')
#scg_frame = loadmat('D:\Onedrive Gatech\OneDrive - Georgia Institute of Technology\Journal and Conferences\Journal on RHC Med Challenge\Data\Features/RHC_PCWP_med_frames_20s_sys_dias_20201203.mat')



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






################plot R^2 between the target variable and DTW Disatances during Systolic Phase ################    
d=[['AX_500ms',calculate_r2(y,dtw_distance_matrix[:,0])],['AY_500ms',calculate_r2(y,dtw_distance_matrix[:,1])],['AZ_500ms', calculate_r2(y,dtw_distance_matrix[:,2])],['AT_500ms', calculate_r2(y,dtw_distance_matrix[:,3])],['AX_IVC',calculate_r2(y,dtw_distance_matrix[:,4])],['AY_IVC',calculate_r2(y,dtw_distance_matrix[:,5])],['AZ_IVC',calculate_r2(y,dtw_distance_matrix[:,6])],['AT_IVC',calculate_r2(y,dtw_distance_matrix[:,7])],['AX_AVC',calculate_r2(y,dtw_distance_matrix[:,8])],['AY_AVC',calculate_r2(y,dtw_distance_matrix[:,9])],['AZ_AVC',calculate_r2(y,dtw_distance_matrix[:,10])],['AT_AVC',calculate_r2(y,dtw_distance_matrix[:,11])]]
df=pd.DataFrame(data=d, columns=['SCG Axes Used','R2'])



plt.figure()
sns.set_style('whitegrid', {'grid.linestyle':'--'})
ax = sns.barplot(x="SCG Axes Used", y="R2", data=df,linewidth=1.5,edgecolor='.2',alpha=0.8, capsize=0.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.set_title('RHC Med challenge\nR^2 between Delta PAMP and DTW Distance for Different SCG Axes',fontsize=12)
plt.tight_layout()
plt.ylim([0.0, 1.0])
#plt.legend(loc="lower right",prop=fontP)
plt.show()



################plot R^2 between VO2 and Euclidean Disatances################    
d=[['AX_Dias_500ms',calculate_r2(y,dtw_distance_matrix[:,12])],['AY_Dias_500ms',calculate_r2(y,dtw_distance_matrix[:,13])],['AZ_Dias_500ms', calculate_r2(y,dtw_distance_matrix[:,14])],['AT_Dias_500ms', calculate_r2(y,dtw_distance_matrix[:,15])],['AX_Dias_active',calculate_r2(y,dtw_distance_matrix[:,16])],['AY_Dias_active',calculate_r2(y,dtw_distance_matrix[:,17])],['AZ_Dias_active',calculate_r2(y,dtw_distance_matrix[:,18])],['AT_Dias_active',calculate_r2(y,dtw_distance_matrix[:,19])],['AX_Dias_passive',calculate_r2(y,dtw_distance_matrix[:,20])],['AY_Dias_passive',calculate_r2(y,dtw_distance_matrix[:,21])],['AZ_Dias_passive',calculate_r2(y,dtw_distance_matrix[:,22])],['AT_Dias_passive',calculate_r2(y,dtw_distance_matrix[:,23])]]
df=pd.DataFrame(data=d, columns=['SCG Axes Used','R2'])



plt.figure()
sns.set_style('whitegrid', {'grid.linestyle':'--'})
ax = sns.barplot(x="SCG Axes Used", y="R2", data=df,linewidth=1.5,edgecolor='.2',alpha=0.8, capsize=0.2)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_linewidth(1)
ax.spines['right'].set_linewidth(1)
ax.spines['top'].set_linewidth(1)
ax.spines['bottom'].set_linewidth(1)
ax.set_title('RHC Med challenge\nR^2 between Delta PAMP and DTW Distance for Different SCG Axes',fontsize=12)
plt.tight_layout()
plt.ylim([0.0, 1.0])
#plt.legend(loc="lower right",prop=fontP)
plt.show()




############Select 5 features using sequential feature selection###########
X=dtw_distance_matrix
X = StandardScaler().fit_transform(X)

reg_model=linear_model.Ridge(alpha=0.5) #### best for Delta PAMP following our initial analysis



sfs1 = SFS(reg_model, 
           k_features=5,  
           forward=True, 
           floating=False, 
           verbose=1,
           scoring='r2',
           n_jobs=-1,
           cv=3)


feature_set=['AX_500ms', 'AY_500ms', 'AZ_500ms', 'AT_500ms','AX_IVC','AY_IVC','AZ_IVC','AT_IVC','AX_EJ','AY_EJ','AZ_EJ','AT_EJ',
             'AX_Dias_500ms','AY_Dias_500ms','AZ_Dias_500ms','AT_Dias_500ms','AX_Dias_active','AY_Dias_active','AZ_Dias_active','AT_Dias_active','AX_Dias_passive','AY_Dias_passive','AZ_Dias_passive','AT_Dias_passive']

sfs1 = sfs1.fit(X, y,custom_feature_names=feature_set)
sfs1.subsets_


####### the following portion of the code is to visualize the feature selection and selected features
fig = plot_sfs(sfs1.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

X_select = sfs1.transform(X)  # select

select_features=sfs1.k_feature_names_
print ('Selected features: ')
print  (select_features)
print ('\n')

X=X_select

####get feature importance/weights ##########
fit=reg_model.fit(X,y)
weights=fit.coef_
print('Feature Weights: ')
print(weights)
print('\n')




###################### Regression ##############
groups=np.unique(groups)
logo = LeaveOneGroupOut()




#initialize vectore to keep score for each fold

y_all_predictions = np.zeros(y.shape)



for train, test in logo.split(X, y, groups=groups):

    #perform train test split for the current fold
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]



#    standardize training and testing data
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0)
#    X_train = (X_train - mean_train)/std_train
#    X_test = (X_test - mean_train)/std_train
    
   
    model=reg_model # regression model is defined before in the feature selection portion

    model_trained = model.fit(X_train , y_train)
    y_predicted = model_trained.predict(X_test)  
    

    y_all_predictions[test] = y_predicted

    # printing the number in output console
    print('Subject No= ' + str(np.mean(groups[test])))
    print('True Delta PAMP = ' + str(np.mean(y_test))+' Predicted Delta PAMP = ' + str(np.mean(y_predicted)))


# calculate RMSE, NRMSE, R-square and pearson p_value between actual and predicted value.
rmse=np.sqrt(mean_squared_error(y, y_all_predictions))
rmse_normalized=rmse/np.mean(y)
r_squared=calculate_r2(y, y_all_predictions)
p_value=sp.stats.pearsonr(y, y_all_predictions)


#print results
print('Overall RMSE= ' + str(rmse))
print('Overall Rsquared= ' + str(r_squared))
print('Pearson Corr-Coeff and P-value= ' + str(p_value))
print('Overvall Normalized RMSE with mean = ' + str(rmse_normalized))


#plot out of fold predictions against actual PEP
fig = plt.figure();
cm=plt.cm.get_cmap('jet')
pl=plt.scatter(y , y_all_predictions , c= groups, cmap= cm)
plt.title('OOF R^2= ' + str(r_squared) +' \nRMSE Normalized with Mean= ' + str(rmse_normalized))
plt.colorbar(pl)
plt.show()


# Bland Altman Plot for the y actual and y predicted
fig = plt.figure();
bland_altman_plot(y, y_all_predictions)
plt.title('Bland-Altman Plot')
plt.ylabel('Actual-Predicted')
plt.xlabel('(Actual+Predicted)/2')
plt.show()


