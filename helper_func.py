import pandas as pd



def groupby_plot(df, groupby_cols, agg_col, title, ylab, xlab, agg_type='mean', plot_type='bar', top_n=10):
    '''
    INPUT 
        df - a dataframe holding the requires columns
        groupby_cols - columns to grouby on
        agg_columns - col to aggregate
        agg_type -  Aggreatgation type (mean, sum, count)
        plot_type - type of chart to return (bar, line)
        top_n -  top n values to display
        title - title of the plot
        ylab - ylabel of the plot
        xlab - xlabel of the plot
    OUTPUT
        agg_sr - a Series of aggregated values
        Displays a plot of the aggregation performed
    '''
    
    if plot_type == 'bar':
        if agg_type == 'mean':
            agg_sr = df.groupby(groupby_cols)[agg_col].mean().sort_values(ascending=False)
            
        if agg_type == 'sum':
            agg_sr = df.groupby(groupby_cols)[agg_col].sum().sort_values(ascending=False)
        agg_sr.head(top_n).plot(kind=plot_type, title=title, ylabel=ylab,xlabel=xlab, figsize=(15, 5))
    elif plot_type == 'line':
        if agg_type == 'mean':
            agg_sr = df.groupby(groupby_cols)[agg_col].mean()
            
        if agg_type == 'sum':
            agg_sr = df.groupby(groupby_cols)[agg_col].sum()
        agg_sr.plot(kind=plot_type, title=title, ylabel=ylab,xlabel=xlab, figsize=(15, 5))

    
    return agg_sr