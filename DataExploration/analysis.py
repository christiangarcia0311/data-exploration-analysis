# Data Exploration Analysis
# Python Data Exploration Analysis
# Library by Christian Garcia

# Import required libraries
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from math import log2
from scipy import stats
from scipy.stats import chi2_contingency, f_oneway

# Univariate Analysis 
class Univariate:

    # --------------- BINNING ------------------

    # Supervised Binning 
    
    # _________ RUNNABLE
    # Entropy Based Binning
    # Calculate entropy fot target
    def calculate_entropy(self, data, target_column):
      total_count = len(data)
      unique_values = data[target_column].unique()
      entropy = 0 
      for values in unique_values:
        probability = len(data[data[target_column] == values]) / total_count
        entropy -= probability * log2(probability)
      
      return entropy
    
    # Calculate informnation gain of the data
    def information_gain(self, data, feature_column, target_column, split_value):
      total_entropy = self.calculate_entropy(data, target_column)
      left_split = data[data[feature_column] <= split_value]
      right_split = data[data[feature_column] > split_value]
      left_entropy = (len(left_split) / len(data)) * self.calculate_entropy(left_split, target_column)
      right_entropy = (len(right_split) / len(data)) * self.calculate_entropy(right_split, target_column)
      info_gain = total_entropy - (left_entropy + right_entropy)
      return info_gain
      
    # Find best split for the given target 
    def best_split(self, data, feature_column, target_column):
      unique_values = sorted(data[feature_column].unique())
      split_value = None
      info_gain = 0 
      for n in range(len(unique_values) - 1):
        splits = (unique_values[n] + unique_values[n + 1]) / 2
        info_gains = self.information_gain(data, feature_column, target_column, splits)
        if info_gains > info_gain:
          info_gain = info_gains
          split_value = splits
          
      return split_value, info_gain
    
    # Display entropy binning 
    def entropy_bin(self, data, feature_column, target_column):
      entropy = self.calculate_entropy(data, target_column)
      split_value, info_gain = self.best_split(data, feature_column, target_column)
      
      data_info = {
        'Entropy': [entropy],
        'Information Gain': [info_gain],
        'Best Split': [split_value]
      }
      info = pd.DataFrame(data_info)
      return info
  
    # display information gain interval
    def split_bins(self, data, feature_column, target_column):
      unique_values = sorted(data[feature_column].unique())
      for n in range(len(unique_values) -1):
        split_value = (unique_values[n] + unique_values[n+1]) / 2
        info_gain = self.information_gain(data, feature_column, target_column, split_value)
        print(f'Bins Split: {split_value} | Information Gain: {info_gain:.3f}')
      
    # Unsupervised Binning
    
    # _________ RUNNABLE
    # Equal Width Based Binning
    def equal_width(self, data, bins=10):
        width = (max(data) - min(data)) / bins
        boundaries = [min(data) + n * width for n in range(1, bins)]
        bin = np.split(np.sort(data), np.searchsorted(np.sort(data), boundaries))
        return bin 

    # Equal Width display bins 
    def ewidth_bin(self, data, bins=10):
        ewidth = self.equal_width(data, bins)
        for n, bin in enumerate(ewidth):
            print(f'Bin {n +1}: {bin}')
    
    # Equal Width Display Bar plot
    def ewidth_plot(self, data, set_title='Equal-Width Based Binning', grid=False, style='ggplot'):
        ewidth1 = [f'{n[0]} - {n[-1]}' for n in data]
        ewidth2 = [sum(n) for n in data]
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        #Display equal width data
        plt.style.use(style)
        plt.bar(ewidth1, ewidth2)
        plt.xlabel('Values')
        plt.ylabel('Width')
        plt.grid(grid)
        plt.show()
    
    # _________ RUNNABLE
    # Equal Frequency Binning
    def equal_freq(self, data, bins=10):
        freq = sorted(data)
        size = len(data) // bins
        boundaries = [freq[n * size] for n in range(1, bins)]
        bin = np.split(freq, np.searchsorted(freq, boundaries))
        return bin

    # Equal Frequency display bins
    def efreq_bin(self, data, bins=10):
        equalfreq = self.equal_freq(data, bins)
        for n, bin in enumerate(equalfreq):
            print(f'Bin {n + 1}: {bin}')

    # Equal Frequency display Bar plot
    def efreq_plot(self, data, set_title='Equal-Frequency Based Binning', grid=False, style='ggplot'):
        equalfreq1 = [f'{n[0]} - n{n[-1]}' for n in data]
        equalfreq2 = [sum(n) for n in data]
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display equal frequency data
        plt.style.use(style)
        plt.bar(equalfreq1, equalfreq2)
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.title(set_title)
        plt.grid(grid)
        plt.show()

    # --------------- ENCODING -------------
    
    # _________ RUNNABLE
    # Binary Encoding
    def binary_encoding_data(self, data, values):
        encode = pd.get_dummies(data[values])
        encoded = encode.astype(int)
        return encoded

    # Binary encoding display with attributes or feature Values
    def binary_encoding(self, data, attribute, values):
        encoded_data = pd.concat([attribute, self.binary_encoding_data(data, values)], axis=1)
        return encoded_data

    # Target-Based Encoding
    def target_encoding(self, data, trend, target):
        data[trend + '_encoded'] = data.groupby(trend)[target].transform('mean')
        show_data = data[[trend, target, 'Encoded']]
        return show_data

    # ------------- MISSING VALUES ----------
    
    # _________ RUNNABLE
    # Imputate Missing Values
    def impute_values(self, data):
        find_missing = data.columns[data.isna().any()].tolist()
        for values in find_missing:
            random_values = data[values].dropna().sample(data[values].isnull().sum(), random_state=0)
            random_values.index = data[data[values].isnull()].index
            data.loc[data[values].isnull(), values] = random_values
        return data 

    # -------------- CATEGORICAL VALUES ------------
    
    # _________ RUNNABLE
    # Values count
    def count(self, data, selected_column):
        counts = data[selected_column].value_counts()
        percent = data[selected_column].value_counts(normalize=True) * 100
        count_info = pd.DataFrame({
          'Count': counts,
          'Count %': percent
        })
        return count_info 

    # Charts categorical values
    
    # _________ RUNNABLE
    # Pie Chart
    def piechart(self, data, column, set_title='Pie Chart - Categorical Values'):
        values, values_count = [], []
        data_values = data[column]
        for n in set(data_values):
            values.append(n)
            values_count.append(data_values.tolist().count(n))
        
        # Display Pie chart
        plt.pie(values_count, labels=values, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title(set_title)
        plt.show()
    
    # _________ RUNNABLE 
    # Bar Chart
    def barchart(self, data, column, set_title='Bar Chart - Categorical Values', grid=False, style='ggplot'):
        series = data[column]
        values_count = series.value_counts()
        values_count.plot(kind='bar')
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display Bar chart
        plt.style.use(style)
        plt.xlabel('Categories')
        plt.ylabel('Counts')
        plt.title(set_title)
        plt.grid(grid)
        plt.show()

    # ------------- NUMERICAL VALUES -------------
    
    # _________ RUNNABLE
    # Determine min, max, mean, median, mode
    def central_tendency(self, data):
        # Iterate the columns to compute the central tendency of each attribute
        results = []
        for column in data.columns:
          data_column = data[column].dropna()
          
          # Compute min, max and mean
          minimum = round(data_column.min(), 3)
          maximum = round(data_column.max(), 3)
          mean = round(data_column.mean(), 3)
          
          # Compute median of the data
          data_sorted = data_column.sort_values().reset_index(drop=True)
          data_length = len(data_sorted)
          if data_length % 2 == 0:
            median = round((data_sorted[data_length // 2 - 1] + data_sorted[data_length // 2]) / 2, 3)
          else:
            median = round(data_sorted[data_length // 2], 3)
          
          # Compute mode of the data
          mode = data_column.mode().tolist()
          mode = [round(n, 3) for n in mode]
          
          results.append({
            'Column': column,
            'Min': minimum,
            'Max': maximum,
            'Mean': mean,
            'Median': median,
            'Mode': mode
          })
        m_stats = pd.DataFrame(results)
        return m_stats
    
    # _________ RUNNABLE
    # Determine range, quantiles, variance, standard deviation and coefficient of variation
    def dispersion_variability(self, data):
        
        # Iterate the columns to compute dispersion/variability of each attribute
        results = []
        for column in data.columns:
          data_column = data[column].dropna()
          
          # Compute range, quantiles, variance, standard deviation and coefficient of variation
          data_range = np.ptp(data_column)
          data_qrtls = np.percentile(data_column, [25, 50, 75])
          data_qrtls = [round(n, 3) for n in data_qrtls]
          data_varis = round(np.var(data_column), 3)
          data_stdev = round(np.std(data_column), 3)
          data_coeff = round((data_stdev / np.mean(data_column)) * 100, 3)
          
          results.append({
            'Column': column,
            'Range': data_range,
            'Q1': data_qrtls[0],
            'Q2': data_qrtls[1],
            'Q3': data_qrtls[2],
            'Variance': data_varis,
            'Standard Deviation': data_stdev,
            'Coefficient of Variation (%)': data_coeff
          })
        data_stats = pd.DataFrame(results)
        return data_stats
    
    # _________ RUNNABLE
    # Determine the skewness and kutosis
    def distribution_shape(self, data):
        results = []
        
        for column in data.columns:
          skewness = stats.skew(data[column].dropna())
          kurtosis = stats.kurtosis(data[column].dropna())
          results.append({
            'Column': column,
            'Skewness': skewness,
            'Kurtosis': kurtosis
          })
        data_stats = pd.DataFrame(results)
        return data_stats
    
    # _________ RUNNABLE    
    # Bar Plots Numerical values 
    def histogram(self, data, bins=10, set_title='Histogram - Numerical Values', grid=False, style='ggplot'):
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display histogram data
        plt.style.use(style)
        plt.hist(data, bins=bins)
        plt.xlabel('Data Values')
        plt.ylabel('Frequency')
        plt.title(set_title)
        plt.grid(grid)
        plt.show()
    
    # _________ RUNNABLE
    # Box Plot Numerical values
    def boxplot(self, data, xlabel='X-axis', ylabel='Y-axis', set_title='Box Plot - Numerical Values', style='ggplot'):
        box = plt.boxplot(data, patch_artist=True)
        colors = ['green', 'orange', 'pink', 'yellow']
        for patch, color in zip(box['boxes'], colors):
          patch.set_facecolor(color)
          
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display boxplot data
        plt.style.use(style)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(set_title)
        plt.show()


# Bivariate Analysis
class Bivariate:
    
    # -------- CATEGORICAL AND CATEGORICAL ----------------
    
    # _________ RUNNABLE
    # Chi Squared test
    def chisquared(self, data):
        chi2, p, dof, expected = chi2_contingency(data)
        results = {
          'Chi-Squared Stats': [round(chi2, 3)],
          'P-value': [round(p, 3)],
          'Degrees of Freedom': [dof]
        }
        chi_squared = pd.DataFrame(results)
        expected_freq = pd.DataFrame(expected, columns=data.columns, index=data.index).round(3)
        return chi_squared, expected_freq
    
    # convert data to contingency table
    def contingency_table(self, data, first_attribute, second_attribute):
      table = pd.crosstab(pd.Series(data[first_attribute], name=first_attribute), pd.Series(data[second_attribute], name=second_attribute))
      return table
    
    # _________ RUNNABLE
    # Bar Chart Numerical values
    def barchart(self, data, first_attribute, second_attribute, mode='vertical', color='#4CAF50', set_title='Bar Chart - Numerical Values', style='ggplot'):
        X = data[first_attribute]
        y = data[second_attribute]
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display bar chart numerical data
        if mode == 'horizontal':
          plt.style.use(style)
          plt.barh(X, y, color=color)
          plt.xlabel(first_attribute)
          plt.ylabel(second_attribute)
          plt.title(set_title)
          plt.show()
        else:
          plt.style.use(style)
          plt.bar(X, y, color=color)
          plt.xlabel(first_attribute)
          plt.ylabel(second_attribute)
          plt.title(set_title)
          plt.show()

    # Stacked Column Chart
    def stacked_column(self, data, first_attribute, second_attribute, set_title='Stacked Column Chart - Numerical Values', style='ggplot'):
        stacked_data = data.groupby([first_attribute, second_attribute]).size().reset_index(name='count')
        stacked_data = stacked_data.pivot(index=second_attribute, columns=first_attribute, values='count').fillna(0)
        stacked_data = stacked_data.div(stacked_data.sum(axis=1), axis=0) * 100
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display stacked column data
        plt.style.use(style)
        stacked_data.plot(kind='bar', stacked=True)
        plt.xlabel(second_attribute)
        plt.ylabel(first_attribute)
        plt.title(set_title)
        plt.show()

    # -------------- NUMERICAL AND NUMERICAL --------------
    
    # _________ RUNNABLE
    # Correlation of the data
    def correlation(self, data, first_attribute, second_attribute, set_title='Linear Correlation - Numerical Values', size=70, style='ggplot'):
        X = data[first_attribute]
        y = data[second_attribute]
        m, b = np.polyfit(X, y, 1)
        N = 3
        colors = np.random.rand(N)
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display Linear correlation
        plt.style.use(style)
        plt.plot(X, m*X+b, color='orange', linestyle='-')
        plt.scatter(X, y, s=size, color=colors)
        plt.xlabel(first_attribute)
        plt.ylabel(second_attribute)
        plt.title(set_title)
        plt.show()
    
    # _________ RUNNABLE
    # Scatterplot relation of two variables
    def scatterplot(self, data, first_attribute, second_attribute, set_title='Scatter Plot - Numerical Values', size=70, style='ggplot'):
        X = data[first_attribute]
        y = data[second_attribute]
        N = 3 
        colors = np.random.rand(N)
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale

        # Display scatterplot
        plt.style.use(style)
        plt.scatter(X, y, s=size, color=colors, alpha=0.5)
        plt.xlabel(first_attribute)
        plt.ylabel(second_attribute)
        plt.title(set_title)
        plt.show()

    # ------------- CATEGORICAL AND NUMERICAL ---------------
     
    # _________ RUNNABLE
    # Z-Test
    def z_test(self, data, first_column, second_column, group_target_X, group_target_Y):
        group_X = data[data[first_column] == group_target_X][second_column]
        group_Y = data[data[first_column] == group_target_Y][second_column]
        
        groupX_mean = np.mean(group_X)
        groupY_mean = np.mean(group_Y)
        groupX_stdev = np.std(group_X)
        groupY_stdev = np.std(group_Y)
        
        length_X = len(group_X)
        length_Y = len(group_Y)
        
        pooled_stdev = np.sqrt(groupX_stdev**2 / length_X + groupY_stdev**2 / length_Y)
        z_statistic = (groupX_mean - groupY_mean) / pooled_stdev
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
        
        z_statistic = round(z_statistic, 3)
        p_value = round(p_value, 3)
        
        ztest_stats = {
          'Z-Statistic': z_statistic,
          'P-value': p_value
        }
        results = pd.DataFrame([ztest_stats])
        return results
    
    # _________ RUNNABLE
    # T-Test
    def t_test(self, data, first_column, second_column, group_target_X, group_target_Y):
        group_X = data[data[first_column] == group_target_X][second_column]
        group_Y = data[data[first_column] == group_target_Y][second_column]
        
        t_statistic, p_value = stats.ttest_ind(group_X, group_Y)
        t_statistic = round(t_statistic, 3)
        p_value = round(p_value, 3)
        ttest_stats = {
          'T-Statistic': t_statistic,
          'P-value': p_value
        }
        results = pd.DataFrame([ttest_stats])
        return results
    
    # _________ RUNNABLE
    # ANOVA
    def anova(self, data, first_column, second_column, group_target_X, group_target_Y):
        group_X = data[data[first_column] == group_target_X][second_column]
        group_Y = data[data[first_column] == group_target_Y][second_column]
        
        f_statistic, p_value = f_oneway(group_X, group_Y)
        f_statistic = round(f_statistic, 3)
        p_value = round(p_value, 3)
        anova_stats = {
          'F-Statistic': f_statistic,
          'P-value': p_value
        }
        results = pd.DataFrame([anova_stats])
        return results
    
    # _________ RUNNABLE
    # Line Chart with error bar
    def linechart(self, data, first_attribute, second_attribute, xlabel='X-Label Axis', ylabel='Y-Label Axis', set_title='Line Chart w/ Error Bar',style='ggplot', size=4):
        data_error = data.groupby(first_attribute)[second_attribute].agg(['mean', 'sem'])
        X_data = np.arange(len(data_error))
        y_data = data_error['mean']
        stdev_error = data_error['sem']
        
        # Pick style plot
        # 1. bmh
        # 2. dark_background
        # 3. fivethirtyeight
        # 4. ggplot
        # 5. grayscale
        
        # Display line chart data
        
        plt.style.use(style)
        fig, axes = plt.subplots()
        axes.errorbar(X_data, y_data, yerr=stdev_error, capsize=size)
        axes.set_xticks(X_data)
        axes.set_xticklabels(data_error.index)
        axes.set_xlabel(xlabel)
        axes.set_ylabel(ylabel)
        axes.set_title(set_title)
        plt.show()
        
    # _________ RUNNABLE
    # 2-Y axis plot
    def y2_axis(self, data, X_axis, y_axis1, y_axis2, set_title='2-Y Axis Plot - Numerical Values', grid=False, style='ggplot'):
      x = data[X_axis]
      y1 = data[y_axis1]
      y2 = data[y_axis2]
      
      # Pick style plot
      # 1. bmh
      # 2. dark_background
      # 3. fivethirtyeight
      # 4. ggplot
      # 5. grayscale
      
      # Display 2-y axis plot data
      plt.style.use(style)
      fig, axes1 = plt.subplots()
      axes2 = axes1.twinx()
      axes1.plot(x, y1, 'g-')
      axes2.plot(x, y2, 'b-')
      
      axes1.set_xlabel(X_axis)
      axes1.set_ylabel(y_axis1)
      axes2.set_ylabel(y_axis2)
      axes1.grid(grid)
      plt.show()

# Sample Datasets applying univariate and bivariate analysis
class Dataset:
  
  # Categorical and Numerical Values
  '''
  Generate dataset with multiple columns in order to fill it with categorical and numerical values.
  '''
  def make_dataset(self, column_details, n_instance=100):
    dataset = {}
    for columns, instance_details in column_details.items():
      if isinstance(instance_details, tuple):
        if instance_details[0] == 'int':
          minimum_value, maximum_value = instance_details[1], instance_details[2]
          dataset[columns] = np.random.randint(minimum_value, maximum_value + 1, n_instance)
        elif instance_details[0] == 'float':
          minimum_value, maximum_value = instance_details[1], instance_details[2]
          dataset[columns] = np.round(np.random.uniform(minimum_value, maximum_value, n_instance), 2)
      else:
        dataset[columns] = np.random.choice(instance_details, n_instance)
    dataset_frame = pd.DataFrame(dataset)
    return dataset_frame