#!/usr/bin/python
# -*- coding: utf-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
import pandas as pd
import statsmodels.api as sm

sample_names = ["fishy_cat", "calafate", "scene-Helicopter-27", "TeeglasFX_27"]
ratio_names = ["calafate", "scene-Helicopter-27", "TeeglasFX_27"]
data = pd.read_csv("linearity_data.csv")
coefficients = DataFrame({},columns=['name', 'multiplier', 'const', 'r-squared', 'linearity_type'])


def LinearitySampleLineDrawing(image_data):
    coefficients = DataFrame({}, index=[str(i) + '_' + str(j) for i in range(4) for j in range(4)],
                             columns=['name', 'multiplier', 'const', 'r-squared', 'linearity_type'])
    for j in range(4):
        for i in range(4):
            temp_data = image_data[image_data['x_position'].isin([i]) & image_data['y_position'].isin([j])]
            x = map(float, temp_data['sample'])
            y = map(float, temp_data['rendering_time'])
            x, y = np.array(x), np.array(y)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y)[0]
            plt.plot(x, y, 'o', label='Original data', markersize=10)
            plt.plot(x, m * x + c, 'r', label='Fitted line: ' + str(round(m, 2)) + '*x+' + str(round(c, 2)))
            plt.legend()
            plt.xlabel('Samples')
            plt.ylabel('Rendering time/s')
            plt.title('Linearity of number of samples for ' + list(temp_data['name'])[0] + ' ' + str(i) + ' ' + str(j))
            plt.savefig(list(temp_data['name'])[0] + '_' + str(i) + '_' + str(j) + '_' + 'sample' + '.png',
                        bbox_inches='tight')
            # plt.show()
            plt.clf()
            model = sm.OLS(y, A)
            results = model.fit()
            coefficients.ix[str(i) + '_' + str(j)] = [list(temp_data['name'])[0], m, c, results.rsquared, 'sample']
    return coefficients


def SampleSyntheticModel(sample_data):
    x = map(float, sample_data['sample'])
    y = map(float, sample_data['rendering_time'])
    x, y = np.array(x), np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, m * x + c, 'r', label='Fitted line: ' + str(round(m, 2)) + '*x+' + str(round(c, 2)))
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Rendering time/s')
    plt.title('Synthetic linear model of number of samples')
    plt.savefig('synthetic_linear_model_of_number_of_samples.png', bbox_inches='tight')
    # plt.show()
    plt.clf()
    model = sm.OLS(y, A)
    results = model.fit()
    return results.rsquared


def RatioTransform(ratio):
    return 1.0 / (ratio ** 2)


def LinearityResolutionLineDrawing(image_data):
    coefficients = DataFrame({}, index=[str(i) + '_' + str(j) for i in range(4) for j in range(4)],
                      columns=['name', 'multiplier', 'const', 'r-squared', 'linearity_type'])
    for j in range(4):
        for i in range(4):
            temp_data = image_data[image_data['x_position'].isin([i]) & image_data['y_position'].isin([j])]
            x = map(float, temp_data['ratio'])
            y = map(float, temp_data['rendering_time'])
            x = map(RatioTransform, x)
            x, y = np.array(x), np.array(y)
            A = np.vstack([x, np.ones(len(x))]).T
            m, c = np.linalg.lstsq(A, y)[0]
            plt.plot(x, y, 'o', label='Original data', markersize=10)
            plt.plot(x, m * x + c, 'r', label='Fitted line: ' + str(round(m, 2)) + '*x+' + str(round(c, 2)))
            plt.legend()
            plt.xlabel('Ratio of contraction')
            plt.ylabel('Rendering time/s')
            plt.title('Linearity of ratio for ' + list(temp_data['name'])[0] + ' ' + str(i) + ' ' + str(j))
            plt.savefig(list(temp_data['name'])[0] + '_' + str(i) + '_' + str(j) + '_' + 'ratio' + '.png',
                        bbox_inches='tight')
            # plt.show()
            plt.clf()
            model = sm.OLS(y, A)
            results = model.fit()
            coefficients.ix[str(i) + '_' + str(j)] = [list(temp_data['name'])[0], m, c, results.rsquared, 'ratio']
    return coefficients


def RatioSytheticLinearModel(ratio_data):
    x = map(float, ratio_data['ratio'])
    y = map(float, ratio_data['rendering_time'])
    x = map(RatioTransform, x)
    x, y = np.array(x), np.array(y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    plt.plot(x, y, 'o', label='Original data', markersize=10)
    plt.plot(x, m * x + c, 'r', label='Fitted line: ' + str(round(m, 2)) + '*x+' + str(round(c, 2)))
    plt.legend()
    plt.xlabel('Ratio of contraction')
    plt.ylabel('Rendering Time/s')
    plt.title('Synthetic linear model of ratio')
    plt.savefig('synthetic_linear_model_of_ratio.png', bbox_inches='tight')
    #plt.show()
    plt.clf()
    model = sm.OLS(y, A)
    results = model.fit()
    return results.rsquared


for name in sample_names:
    image_data = data[data['name'].isin([name]) & data['ratio'].isin([1])]
    if sample_names.index(name) == 0:
        coefficients = LinearitySampleLineDrawing(image_data)
    else:
        coefficients = coefficients.append(LinearitySampleLineDrawing(image_data))
for name in ratio_names:
    image_data = data[data['name'].isin([name]) & data['sample'].isin([2000])]
    coefficients = coefficients.append(LinearityResolutionLineDrawing(image_data))
coefficients.to_csv('linear_model_coefficients.csv',index=False)
r_squared = []
sample_data = data[data['ratio'].isin([1])]
r_squared.append(SampleSyntheticModel(sample_data))
ratio_data = data[data['sample'].isin([2000])]
r_squared.append(RatioSytheticLinearModel(ratio_data))
r_squared=Series(r_squared,index=["R-squared of sample","R-squared of ratio"])
r_squared.to_csv("synthetic_model_r_squared.csv")