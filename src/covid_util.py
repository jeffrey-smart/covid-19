import logging
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import sys



def get_covid_data_jhu(category, region='US'):
    ''' Get COVID-19 data from Johns Hopkins University; return DataFrame.'''

    # pd.read_csv() will raise exception if the file name is not valid,
    # this assert statement provides valid tags (as of 2020-05-19)
    p = category in {'confirmed', 'deaths'}
    assert p, print(f'category {category} not recognized')
    
    covid_csv_file = (
        'https://raw.githubusercontent.com'
        '/CSSEGISandData'
        '/COVID-19'
        '/master'
        '/csse_covid_19_data'
        '/csse_covid_19_time_series'
        f'/time_series_covid19_{category}_{region}.csv'
    )
    t = pd.read_csv(covid_csv_file)
    logging.info('function %s | covid_file = %s', 'get_covid_data_jhu', covid_csv_file)
    logging.info('function %s | category %s | shape %r', 'get_covid_data_jhu', category, t.shape)
    
    return t

def reshape(df, category):
    ''' Convert dataframe to tidy format.
    
        Input:  1 row for each state+county; 1 column for each date
        
        Output: 1 row for each date (as index, with type datetime64[D]); 
                1 column for each state+county (as multi-index)
    '''

    date_idx = df.columns.get_loc('1/22/20')

    t = (df.melt(id_vars = ['Province_State', 'Admin2'],
                 value_vars = df.columns[date_idx : ],
                 var_name = 'date',
                 value_name = category,
                )
         .assign(date = lambda x: pd.to_datetime(x['date'], 
                                                 format='%m/%d/%y', 
                                                 errors='coerce'))
         .set_index(['date', 'Province_State', 'Admin2'], 
                    verify_integrity=True)
         .sort_index()
         .squeeze()
         .unstack(level=['Province_State', 'Admin2'])
        )
    
    # we use `.sort_index()` so don't need these `assert` statements
    assert t.index.is_monotonic_increasing
    assert t.columns.is_monotonic_increasing
    logging.info('function %s | category %s | shape %r', 'reshape', category, t.shape)
    
    return t

def extract_population(df):
    
    ''' Extract US population data from the `deaths` dataframe.
    
        Input:  dataframe with US deaths (and population!)
        Output: pandas series with population (multi-index by State, County)
    '''
    
    t = (df
         .filter(['Province_State', 'Admin2', 'Population'])
         .set_index(['Province_State', 'Admin2'], verify_integrity=True)
         .sort_index()
         .squeeze()
        )
    
    logging.info('function %s | shape %r', 'extract_population', t.shape)
    logging.info('status : done extracting population')

    return t

class Error(Exception):
    ''' Base class for exceptions in this module.'''
    pass

class DataValidationError(Error):
    def __init__(self, message):
        self.message = message


def initial_validation(dfs, pop, categories):

    # validation
    
    p = set(dfs.keys()) == set(categories)
    if not p:
        raise DataValidationError('dict keys != categories')

    p = (dfs['confirmed'].index == dfs['deaths'].index).all()
    if not p:
        raise DataValidationError('confirmed index != deaths index')
    
    p = (dfs['confirmed'].columns == dfs['deaths'].columns).all()
    if not p:
        raise DataValidationError('confirmed columns != deaths columns')

    p = (dfs['confirmed'].columns == pop.index).all()
    if not p:
        raise DataValidationError('confirmed columns != population index')

    logging.info('function %s | status : all initial data consistency checks passed', 
                 'initial_validation')
    
    return True


def latest_date_state_level(dfs, pop):
    ''' Create summary table for latest available date.'''
    
    # population: create state-level summary (not state + county)
    state_pop = pop.sum(level='Province_State')

    # covid confirmed cases + deaths (latest available date)
    ts = list()
    for category in dfs.keys():
        cases = dfs[category].iloc[-1].sum(level='Province_State').rename(category)
        ts.append(cases)
        
        per_100k = cases.div(state_pop).mul(100_000).rename(category + ' per 100k')
        ts.append(per_100k)

    ts.append(state_pop)
    
    # several pandas series => one dataframe
    frame = pd.concat(ts, axis=1)

    frame['as_of_date'] = dfs[category].index.max()

    # exclude the two cruise ships
    mask = frame.index.isin({'Diamond Princess', 'Grand Princess'})
    frame = frame[~ mask]

    return frame



state_name_to_code = {
    'California': 'CA',
    'Connecticut': 'CT',
    'Florida': 'FL',
    'Illinois': 'IL',
    'Louisiana': 'LA',
    'Maryland': 'MA',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'New Jersey': 'NJ',
    'New York': 'NY',
    'Pennsylvania': 'PA',
    'Texas': 'TX',
}



def plot_count_vs_rate(data, category):
    fig, ax = plt.subplots(figsize=(10, 6))

    as_of = data['as_of_date'].iloc[0].strftime('%Y-%m-%d')
    x = category
    y = f'{category} per 100k'
    
    ax.scatter(data[x].values, data[y].values, alpha=0.5)

    ax.set(xlabel = x.title(),
          ylabel = y.title(),
          title = f'Number of {x.title()} vs. Number of {y.title()} as of {as_of}',
          )
    
    # add annotation for top 10 items
    for state in data[x].sort_values(ascending=False).index[0:10]:
        state_x = data.at[state, x]
        state_y = data.at[state, y]

        ax.annotate(state_name_to_code.get(state, state),
                    xy=(state_x, state_y),
                    xycoords='data',
                    xytext=(5, 0),
                    textcoords='offset points',
                    #arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='left',
                    verticalalignment='center')

    return fig, ax


def plot_confirmed_vs_deaths(data):

    fig, ax = plt.subplots(figsize=(10, 6))
    
    as_of = data['as_of_date'].iloc[0].strftime('%Y-%m-%d')

    mask = data['deaths'] >= 10

    ax.scatter(data.loc[mask, 'confirmed'].values,
               data.loc[mask, 'deaths'].values,
               alpha=0.5)

    ax.set(xlabel='Number of Confirmed Cases',
           ylabel='Number of Deaths',
           xscale='log',
           yscale='log',
           title=f'Number of Confirmed Cases vs Number of Deaths as of {as_of}',
          )

    ax.annotate('for states reporting\n10 or more deaths',
                xy=(1, 0),
                xycoords='axes fraction',
                xytext=(-20, 20),
                textcoords='offset pixels',
                horizontalalignment='right',
                verticalalignment='bottom',
               )
    
    # add annotation for top 10 items
    x = 'confirmed'
    y = 'deaths'
    
    for state in data[y].sort_values(ascending=False).index[0:10]:
        state_x = data.at[state, x]
        state_y = data.at[state, y]

        ax.annotate(state_name_to_code.get(state, state),
                    xy=(state_x, state_y),
                    xycoords='data',
                    xytext=(5, 0),
                    textcoords='offset points',
                    #arrowprops=dict(facecolor='black', shrink=0.05),
                    horizontalalignment='left',
                    verticalalignment='center')

    
    return fig, ax


def plot_observations_vs_date(data, category):

    fig, ax = plt.subplots(figsize=(10, 6))

    top_10_states = (data[category]
                     .sum(axis=1, level='Province_State')
                     .iloc[-1]
                     .sort_values(ascending=False)
                     .index[0:10]
                    )

    for state in top_10_states:
        t = data[category].sum(axis=1, level='Province_State').loc[:, state].loc[lambda x: x >= 10]
        ax.plot(t, label=state)

    ax.set(xlabel='Date',
           ylabel=f'Number of {category.title()}',
           yscale='log',
           title=f'Number of {category.title()}'
          )
    ax.get_xaxis().set_major_locator(mdates.DayLocator(interval=14))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%b %d'))

    ax.legend()

    return fig, ax

def plot_observations_vs_days(data, category):

    fig, ax = plt.subplots(figsize=(10, 6))

    top_10_states = (data[category]
                     .sum(axis=1, level='Province_State')
                     .iloc[-1]
                     .sort_values(ascending=False)
                     .index[0:10]
                    )

    for state in top_10_states:
        t = (data[category]
             .sum(axis=1, level='Province_State')
             .loc[:, state]
             .loc[lambda x: x >= 10]
             .reset_index(drop=True)
            )
        ax.plot(t, label=state)

    ax.set(xlabel=f'Days since 10th {category.title()}',
           ylabel=f'Number of {category.title()}',
           yscale='log',
           title=f'Number of {category.title()}'
          )

    ax.legend()

    return fig, ax

# counties in Southern California (So Cal)
so_cal_counties = [
    'Imperial',
    'Kern',
    'Los Angeles',
    'Orange',
    'Riverside',
    'San Bernardino',
    'San Diego',
    'San Luis Obispo',
    'Santa Barbara',
    'Ventura',
    ]

def drop_level_inline(df):
    t = df.copy()
    t.columns = t.columns.droplevel(0)
    return t

def confirmed_cases_so_cal(data, category):
    fig, ax = plt.subplots(figsize=(10, 6))

    t = (data[category]
         .loc[:, ('California', so_cal_counties)]
         .pipe(drop_level_inline)
        )

    for county in t.columns:
        ax.plot(t[county].loc[lambda x: x >= 10], label=county)

    ax.set(xlabel='date',
           ylabel='# of Confirmed Cases',
           yscale='log',
           title='Confirmed Cases in Southern California',
          )

    ax.get_xaxis().set_major_locator(mdates.DayLocator(interval=14))
    ax.get_xaxis().set_major_formatter(mdates.DateFormatter('%b %d'))

    ax.legend()
    
    return fig, ax
