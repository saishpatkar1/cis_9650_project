#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 19:34:02 2020

@author: spatkar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


#Read csv and get data

df = pd.read_csv("file:///Users/spatkar/Desktop/05b_analysis_file_update.csv")


#SafetyNet Variables

change_in_econ_sn_eitc_irs_share_values_2013_to_2017 = df.groupby(['cocnumber'])['econ_sn_eitc_irs_share'].mean()

share_of_households_with_public_assistance_income_2016 = df.groupby(['cocnumber'])['econ_sn_cashasst_acs5yr_2017'].mean()

percentage_of_housing_units_2016_built_before_1940 = df.groupby(['cocnumber'])['hou_mkt_homeage1940_acs5yr_2017'].mean()

share_HUD_subsidized_housing_to_total = df.groupby(['cocnumber'])['hou_pol_hudunit_psh_hud_share'].mean()

share_of_under_18_population_living_with_single_parent = df.groupby(['cocnumber'])['dem_soc_singparent_acs5yr'].mean()

share_of_the_civilian_population_25_and_older_with_veteran_status  = df.groupby(['cocnumber'])['dem_soc_vet_acs5yr'].mean()


share_of_households_with_public_assistance_income_2016.to_csv("share_of_households_with_public_assistance_income_2016.csv")

change_in_econ_sn_eitc_irs_share_values_2013_to_2017.to_csv("change_in_econ_sn_eitc_irs_share_values_2013_to_2017.csv")

percentage_of_housing_units_2016_built_before_1940.to_csv("percentage_of_housing_units_2016_built_before_1940.csv")

share_HUD_subsidized_housing_to_total.to_csv("share_HUD_subsidized_housing_to_total.csv")



#Demographics Variables

education_bachelors_or_higher_to_total_population_ages_25_64   = df.groupby(['cocnumber'])['dem_soc_ed_bach_acs5yr'].mean()

total_white_population = df.groupby(['cocnumber'])['dem_soc_white_census'].mean()

total_population = df.groupby(['cocnumber'])['dem_pop_pop_census'].mean()

ratio_white_population_to_total_population = total_white_population/total_population

total_black_population = df.groupby(['cocnumber'])['dem_soc_black_census'].mean()


ratio_black_population_to_total_population = total_black_population/total_population


ratio_black_population_to_total_population = total_black_population/total_population

total_hispanic_population = df.groupby(['cocnumber'])['dem_soc_hispanic_census'].mean()


ratio_hispanic_population_to_total_population = total_hispanic_population/total_population


total_asian_population = df.groupby(['cocnumber'])['dem_soc_asian_census'].mean()

ratio_asian_population_to_total_population = total_asian_population/total_population


total_other_population = df.groupby(['cocnumber'])['dem_soc_racetwo_census'].mean()

ratio_other_population_to_total_population = total_other_population/total_population

share_of_under_18_population_living_with_single_parent.to_csv('share_of_under_18_population_living_with_single_parent.csv')

share_of_the_civilian_population_25_and_older_with_veteran_status.to_csv('share_of_the_civilian_population_25_and_older_with_veteran_status.csv')

education_bachelors_or_higher_to_total_population_ages_25_64.to_csv('education_bachelors_or_higher_to_total_population_ages_25_64.csv')

ratio_white_population_to_total_population.to_csv('ratio_white_population_to_total_population.csv')

combined_demographics2 = pd.DataFrame() 

combined_demographics2['ratio_black_population_to_total_population']= ratio_black_population_to_total_population

combined_demographics2['ratio_asian_population_to_total_population']= ratio_asian_population_to_total_population
combined_demographics2['ratio_other_population_to_total_population ']= ratio_other_population_to_total_population 

combined_demographics2['ratio_hispanic_population_to_total_population'] = ratio_hispanic_population_to_total_population

combined_demographics2.to_csv('combined_demographics2.csv')


#Housing Variables


HousingVariables = df[['year', 'cocnumber', 'hou_mkt_medrent_acs5yr', 'hou_mkt_rentvacancy_acs5yr', 'hou_mkt_utility_acs5yr', 'hou_mkt_burden_sev_rent_acs5yr']]
HousingGrouped = HousingVariables.groupby(['cocnumber'])[['hou_mkt_medrent_acs5yr', 'hou_mkt_rentvacancy_acs5yr', 'hou_mkt_utility_acs5yr', 'hou_mkt_burden_sev_rent_acs5yr']].agg('mean').fillna(HousingVariables.mean()).reset_index() 

HousingGrouped.to_csv("combined_Housing.csv")


#Economics Variables

ecoVariables = df[['year', 'cocnumber', 'econ_labor_incineq_acs5yr', 'econ_labor_unemp_pop_BLS', 'econ_labor_medinc_acs5yr', 'econ_labor_pov_pop_census_share']]
ecoGrouped = ecoVariables.groupby(['cocnumber'])[['econ_labor_incineq_acs5yr', 'econ_labor_unemp_pop_BLS', 'econ_labor_medinc_acs5yr', 'econ_labor_pov_pop_census_share']].agg('mean').fillna(ecoVariables.mean()).reset_index()

ecoGrouped.to_csv("combined_EcoGroup.csv")



#Climate Variables


climateVariables = df[['year', 'cocnumber', 'env_wea_avgtemp_noaa', 'env_wea_avgtemp_summer_noaa', 'env_wea_precip_noaa', 'env_wea_precip_annual_noaa']]
climateGrouped = climateVariables.groupby(['cocnumber'])[['env_wea_avgtemp_noaa', 'env_wea_avgtemp_summer_noaa', 'env_wea_precip_noaa', 'env_wea_precip_annual_noaa']].agg('mean').fillna(ecoVariables.mean()).reset_index()

climateGrouped.to_csv("combined_climateGroup.csv")



#Graphing based on homelessness per 10000 by Cocnumber

homelessness_per_10000_by_cocnumber = df.groupby('cocnumber')['pit_hless_pit_hud_share'].mean()
homelessness_per_10000_by_cocnumber.plot()
plt.show()




#Used Seaborn to Plot Sheltered, Unsheltered, Homeless by Year

totalsGrouped = df.groupby('year')[['pit_tot_hless_pit_hud','pit_tot_unshelt_pit_hud','pit_tot_shelt_pit_hud']].agg('sum').reset_index()

melted = totalsGrouped.melt('year', var_name='Outcomes', value_name='Totals')


plt.figure(figsize=(10,6))
sns.set_style('whitegrid')
sns.set_context('notebook', font_scale=1, rc={'grid.linewidth':2.5})
sns.set_palette('bright')

sns.catplot(x = 'year', y = 'Totals', hue = 'Outcomes', kind='bar', data=melted, ci=None)
plt.title('Total Outcomes By Year')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import os
import matplotlib.pylab as plt
from pandas.plotting import scatter_matrix, parallel_coordinates
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import graphviz
import pydotplus
from sklearn.metrics import mean_squared_error
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import datasets, linear_model
from scipy import stats
from sklearn import metrics



df = pd.read_csv("master.csv")



df.dtypes



df.describe().tail()


#check for missing values

df.isnull().sum()




# check for duplicate values

df.duplicated().sum()




#correlation matrix...predictors should not be too highly correlated to avoid multicollinearity
dfcorr = df.drop(columns = ['cocnumber'])
corrMatrix = dfcorr.corr()
plt.figure(figsize=(12,8))
ax = sns.heatmap(corrMatrix, annot=True, linewidths=.5)




dfcorr.corr()




#check the distribution of the dependent variable
sns.distplot(df['totHLess'])



#apply a transformation to alleviate highly skewed dependent variable
df['totHLesslog'] = np.log(df['totHLess'])



sns.distplot(df['totHLesslog'])



#check that new log transformed variable was created
df.head()




#create training and validation set 

x = df.drop(columns=['totHLess','totHLesslog', 'cocnumber'])
y = df['totHLess']

trainx, validx, trainy, validy = train_test_split(x, y, test_size=0.4, random_state=1)

print(trainx.shape, trainy.shape)
print(validx.shape, validy.shape)




#multiple linear regression on non-transformed dependent variable

home_lm = LinearRegression()
home_lm.fit(trainx, trainy)

coeff_df = pd.DataFrame(home_lm.coef_,x.columns,columns=['Coefficient'])
coeff_df




predictions = home_lm.predict(validx)



#plot actual vs. predicted

plt.scatter(validy, predictions)
plt.xlabel('TrueValues')
plt.ylabel('Predictions')



#calulcate R2 for model

val_set_r2 = r2_score(validy, predictions)

print(val_set_r2)


# ## Multiple Linear Regression on Log Transformed Dependent Variable



x = df.drop(columns=['totHLess','totHLesslog', 'cocnumber'])
y = df['totHLesslog']

trainx, validx, trainy, validy = train_test_split(x, y, test_size=0.4, random_state=1)

print(trainx.shape, trainy.shape)
print(validx.shape, validy.shape)




home_lm = LinearRegression()
home_lm.fit(trainx, trainy)

coeff_df = pd.DataFrame(home_lm.coef_,x.columns,columns=['Coefficient'])
coeff_df.sort_values(['Coefficient'])




predictions = home_lm.predict(validx)




plt.scatter(validy, predictions)
plt.xlabel('TrueValues')
plt.ylabel('Predictions')




val_set_r2 = r2_score(validy, predictions)
print(val_set_r2)




#get absolute values for coefficients

coeff_df.Coefficient = coeff_df.Coefficient.abs()
coeff_df.sort_values(['Coefficient'])





