import streamlit as st
from requests_html import HTMLSession
import pandas as pd
from os import listdir
from os.path import join, isfile

def main():
    ## main streamlit app to host on heroku
    
    dfs = ['defenders_bundesliga.csv', 'forwards_bundesliga.csv', 
           'forwards_epl.csv', 'forwards_ligue1.csv', 'midfielders_epl.csv', 
           'midfielders_ligue1.csv']

    data = pd.DataFrame(columns = ['№', 'Season', 'Team', 'Apps', 'Min', 'G', 'A', 'Sh90', 'KP90', 'xG',
       'xA', 'xG90', 'xA90', 'Player', 'League', 'Pos'])
    for file in dfs:
        temp = pd.read_csv(file)
        temp.drop('Unnamed: 0', axis = 1, inplace = True)
        data = data.append(temp)

    data.dropna(axis = 0, inplace = True)
    data.reset_index(inplace = True)
    data.drop(['index','№'], axis = 1, inplace = True)
    data['Season'] = pd.Categorical(data['Season'], categories = ['2014/2015', '2015/2016', '2016/2017', '2017/2018', '2018/2019',
       '2019/2020'], ordered = True)
    data.set_index(['Player', 'Season'], inplace = True)

    import re

    def delta(x):
        x = str(x)
        if len(x) >= 5:
            if x[-5] == '+':
                x = x.replace('+', ' +')
            elif x[-5] == '-':
                x = x.replace('-', ' -')
            else:
                x = x
        else:
            x = x
        return x 

    def split(x):
        if len(x.split()) == 2:
            x, y = x.split()
        else:
            y = 0
        return x, y
    
     
    data['xG'] = data['xG'].apply(lambda x: delta(x))
    data['xA'] = data['xA'].apply(lambda x: delta(x))

    x = data['xG'].apply(lambda x: split(x))
    data['xG'] = [x[0] for x in x]
    data['deltaXG'] = [x[1] for x in x]
    y = data['xA'].apply(lambda x: split(x))
    data['xA'] = [x[0] for x in y]
    data['deltaXA'] = [x[1] for x in y]
    data['Min'] = data['Min'] / 1000

    data[['Apps', 'G', 'A']] = data[['Apps', 'G', 'A']].astype(int)

    data[['Min','xG', 'xA', 'deltaXG', 'deltaXA']] = data[['Min','xG', 'xA', 'deltaXG', 'deltaXA']].astype(float)


    transfers = ['David Alaba', 'Raúl Jiménez','Ismaila Sarr', 'Houssem Aouar', 'Victor Osimhen', 
            'Edinson Cavani', 'Jack Grealish', 'Adama Traoré','Kai Havertz']

    ###Plotting performance pentagons
    ### Min , Sh90, KP90, xG, xA, XG90, XA90, deltaXG, deltaXA 

    import numpy as np
    from bokeh.layouts import widgetbox, column, row
    from bokeh.models import ColumnDataSource, Select, LabelSet, Patches, Button, Range1d, FactorRange, Segment, Circle, Plot, LinearAxis
    from bokeh.plotting import figure, show, output_file
    from bokeh.models.callbacks import CustomJS
    from bokeh.palettes import RdBu3, Blues8
    from bokeh.transform import factor_cmap, cumsum, linear_cmap
    from bokeh.models.annotations import Title

    metrics = 7

    centre = 0.5

    theta = np.linspace(0, 2*np.pi, metrics, endpoint= False)
    # rotate theta such that the first axis is at the top
    theta += np.pi/2

    def unit_poly_verts(theta, centre, sc):
        """Return vertices of polygon for subplot axes.
        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
        """
        x0, y0 = [centre ] * 2
        r = sc
        verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
        return verts

    def radar_patch(r, theta, centre,scale):
        """ Returns the x and y coordinates corresponding to the magnitudes of 
        each variable displayed in the radar plot
        """
        # offset from centre of circle
        offset = 0.01
        scaled = (r / scale)
        yt = (scaled * centre + offset) * np.sin(theta) + centre 
        xt = (scaled* centre + offset) * np.cos(theta) + centre 
    
        return yt, xt

    verts = unit_poly_verts(theta, centre, sc = .5)
    x = [v[0] for v in verts] + [centre]
    y = [v[1] for v in verts] + [1]

    verts = unit_poly_verts(theta, centre, sc = .375)
    y75 = [v[1] for v in verts] 
    y75 = y75 + [y75[0]]
    x75 = [v[0] for v in verts] 
    x75 = x75 + [x75[0]]

    verts = unit_poly_verts(theta, centre, sc = .25)
    y50 = [v[1] for v in verts] 
    y50 = y50 + [y50[0]]
    x50 = [v[0] for v in verts] 
    x50 = x50 + [x50[0]]

    verts = unit_poly_verts(theta, centre, sc = .125)
    y25 = [v[1] for v in verts] 
    y25 = y25 + [y25[0]]
    x25 = [v[0] for v in verts] 
    x25 = x25 + [x25[0]]


    from bokeh.models import HoverTool
    hover = HoverTool(tooltips = None, mode = 'vline')
    title = 'Season 2019/2020'
    p = figure(title="Season 2019/2020 ", plot_width=630, plot_height = 510  ,toolbar_location=None, x_range= [-.15,1.2])
    text = ['Min', 'xA90', 'xG90', 'xA', 'xG', 'KP90', 'Sh90', '']

    ### labels scales p.multi_lines
    scale1 = data[data['Min'] >=.5].groupby(['League','Pos']).max()[['Min','Sh90','KP90','xG','xA','xG90','xA90']].reset_index()
 

    # Jack Grealish
    scale_75 = scale1.loc[3][2:] * .75
    labels75_1 = [f"{num:.1f}" for num in scale_75]
    item1 = labels75_1[0]
    labels75_1.reverse()
    labels75_1 = [item1] + labels75_1[0:6] + ['']
    scale_50 = scale1.loc[3][2:] * .5
    labels50_1 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_1[0]
    labels50_1.reverse()
    labels50_1 = [item1] + labels50_1[0:6] + ['']
    scale_25 = scale1.loc[3][2:] * .25
    labels25_1 = [f"{num:.1f}" for num in scale_25]
    item1 = labels25_1[0]
    labels25_1.reverse()
    labels25_1 = [item1] + labels25_1[0:6] + ['']

    # Adama Traore # Ismaila Sarr # Raul Jimenez
    scale_75 = scale1.loc[2][2:] * .75
    labels75_2 = [f"{num:.1f}" for num in scale_75] 
    item1 = labels75_2[0]
    labels75_2.reverse()
    labels75_2 = [item1] + labels75_2[0:6] + ['']
    scale_50 = scale1.loc[2][2:] * .5
    labels50_2 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_2[0]
    labels50_2.reverse()
    labels50_2 = [item1] + labels50_2[0:6] + ['']
    scale_25 = scale1.loc[2][2:] * .25
    labels25_2 = [f"{num:.1f}" for num in scale_25] 
    item1 = labels25_2[0]
    labels25_2.reverse()
    labels25_2 = [item1] + labels25_2[0:6] + ['']


    # Houssem
    scale_75 = scale1.loc[5][2:] *.75
    labels75_3 = [f"{num:.1f}" for num in scale_75] 
    item1 = labels75_3[0]
    labels75_3.reverse()
    labels75_3 = [item1] + labels75_3[0:6] + ['']
    scale_50 = scale1.loc[5][2:] *.5
    labels50_3 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_3[0]
    labels50_3.reverse()
    labels50_3 = [item1] + labels50_3[0:6] + ['']
    scale_25 = scale1.loc[5][2:] *.25
    labels25_3 = [f"{num:.1f}" for num in scale_25] 
    item1 = labels25_3[0]
    labels25_3.reverse()
    labels25_3 = [item1] + labels25_3[0:6] + ['']


    # Victor  # Edison

    scale_75 = scale1.loc[4][2:] *.75
    labels75_4 = [f"{num:.1f}" for num in scale_75]
    item1 = labels75_4[0]
    labels75_4.reverse()
    labels75_4 = [item1] + labels75_4[0:6] + ['']
    scale_50 = scale1.loc[4][2:] *.5
    labels50_4 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_4[0]
    labels50_4.reverse()
    labels50_4 = [item1] + labels50_4[0:6] + ['']
    scale_25 = scale1.loc[4][2:] *.25
    labels25_4 = [f"{num:.1f}" for num in scale_25] 
    item1 = labels25_4[0]
    labels25_4.reverse()
    labels25_4 = [item1] + labels25_4[0:6] + ['']

    # Kai
    scale_75 = scale1.loc[1][2:] *.75
    labels75_5 = [f"{num:.1f}" for num in scale_75]
    item1 = labels75_5[0]
    labels75_5.reverse()
    labels75_5 = [item1] + labels75_5[0:6] + ['']
    scale_50 = scale1.loc[1][2:] *.5
    labels50_5 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_5[0]
    labels50_5.reverse()
    labels50_5 = [item1] + labels50_5[0:6] + ['']
    scale_25 = scale1.loc[1][2:] *.25
    labels25_5 = [f"{num:.1f}" for num in scale_25] 
    item1 = labels25_5[0]
    labels25_5.reverse()
    labels25_5 = [item1] + labels25_5[0:6] + ['']


    # David
    scale_75 = scale1.loc[0][2:] *.75
    labels75_6 = [f"{num:.1f}" for num in scale_75]
    item1 = labels75_6[0]
    labels75_6.reverse()
    labels75_6 = [item1] + labels75_6[0:6] + ['']
    scale_50 = scale1.loc[0][2:] *.5
    labels50_6 = [f"{num:.1f}" for num in scale_50]
    item1 = labels50_6[0]
    labels50_6.reverse()
    labels50_6 = [item1] + labels50_6[0:6] + ['']
    scale_25 = scale1.loc[0][2:] *.25
    labels25_6 = [f"{num:.1f}" for num in scale_25]
    item1 = labels25_6[0]
    labels25_6.reverse()
    labels25_6 = [item1] + labels25_6[0:6] + ['']

    # Ferran


    ## x and y
    df_lab = pd.DataFrame(data = {'x_cor' : [x, x75, x50, x25] ,'y_cor' : [ y, y75, y50, y25]})

    ## text labels
    df1_lab = pd.DataFrame(data = {'text' : [text, labels75_1,labels50_1,labels25_1]})
    df2_lab = pd.DataFrame(data = {'text' : [text, labels75_2,labels50_2,labels25_2]})
    df3_lab = pd.DataFrame(data = {'text' : [text, labels75_3,labels50_3,labels25_3]})
    df4_lab = pd.DataFrame(data = {'text' : [text, labels75_4,labels50_4,labels25_4]})
    df5_lab = pd.DataFrame(data = {'text' : [text, labels75_5,labels50_5,labels25_5]})
    df6_lab = pd.DataFrame(data = {'text' : [text, labels75_6,labels50_6,labels25_6]})



    label_xy = (list(df_lab['x_cor'].values), list(df_lab['y_cor']))
    label_1 = list(df1_lab['text'].values)
    label_2 = list(df2_lab['text'].values)
    label_3 = list(df3_lab['text'].values)
    label_4 = list(df4_lab['text'].values)
    label_5 = list(df5_lab['text'].values)
    label_6 = list(df6_lab['text'].values)

    labels_scale = {
        'x':x,
        'x1':label_xy[0][1],
        'x2':label_xy[0][2],
        'x3':label_xy[0][3],
        'y':y,
        'y1': label_xy[1][1],
        'y2': label_xy[1][2],
        'y3': label_xy[1][3],
        'text': label_1[0],
        'text1': label_1[1],
        'text2': label_1[2],
        'text3': label_1[3],
        'Jack Grealish_l' : label_1[1] ,
        'Jack Grealish_l1' : label_1[2] ,
        'Jack Grealish_l2' : label_1[3] ,
        'Adama Traoré_l' : label_2[1] ,
        'Adama Traoré_l1' : label_2[2] ,
        'Adama Traoré_l2' : label_2[3] ,
        'Ismaila Sarr_l' : label_1[1],
        'Ismaila Sarr_l1' : label_1[2],
        'Ismaila Sarr_l2' : label_1[3],
        'Raúl Jiménez_l' : label_2[1],
        'Raúl Jiménez_l1' : label_2[2],
        'Raúl Jiménez_l2' : label_2[3],
        'Houssem Aouar_l' : label_3[1],
        'Houssem Aouar_l1' : label_3[2],
        'Houssem Aouar_l2' : label_3[3],
        'Victor Osimhen_l' : label_4[1],
        'Victor Osimhen_l1' : label_4[2],
        'Victor Osimhen_l2' : label_4[3],
        'Edison Cavani_l' : label_4[1],
        'Edison Cavani_l1' : label_4[2],
        'Edison Cavani_l2' : label_4[3],
        'Kai Havertz_l' : label_5[1],
        'Kai Havertz_l1' : label_5[2],
        'Kai Havertz_l2' : label_5[3],
        'David Alaba_l' : label_6[1],
        'David Alaba_l1' : label_6[2],
        'David Alaba_l2' : label_6[3]
    }

    lines_src = ColumnDataSource(labels_scale)

    fsize = 5

    p.line(x="x", y="y", source=lines_src)

    labels = LabelSet(x="x",y="y",text="text",source=lines_src)

    p.add_layout(labels)

    a = p.line(x = 'x1', y = 'y1', source = lines_src)

    labels_1 = LabelSet(x="x1",y="y1",text="text1",source=lines_src, text_color = 'grey')

    p.add_layout(labels_1)

    b = p.line(x = 'x2', y = 'y2', source = lines_src)

    labels_2 = LabelSet(x="x2",y="y2",text="text2",source=lines_src, text_color = 'grey')

    p.add_layout(labels_2)

    c = p.line(x = 'x3', y = 'y3', source = lines_src)

    labels_3 = LabelSet(x="x3",y="y3",text="text3",source=lines_src, text_color = 'grey')

    p.add_layout(labels_3)

    a.glyph.line_color = "grey"
    a.glyph.line_alpha = .6
    b.glyph.line_color = "grey"
    b.glyph.line_alpha = .6
    c.glyph.line_color = "grey"
    c.glyph.line_alpha = .6

    ### Polygons
    # example factor:
    theta = np.linspace(0, 2*np.pi, metrics, endpoint= False)
    scale = data[data['Min'] >=.5].groupby(['League','Pos']).max()[['Min','Sh90','KP90','xG','xA','xG90','xA90']].reset_index()
    df_def = data[data['Min'] >= .5].groupby(['League','Pos']).mean()[['Min','Sh90','KP90','xG','xA','xG90','xA90']].reset_index()   
    src_ = data.loc[transfers,'2019/2020',:].iloc[[0,1,2,3,4,5,6,8,11]].reset_index()
    # Jack Grealish
    scale_ = scale.loc[3][2:]
    default_init = df_def.loc[3][2:]

    f1 = src_.loc[6][[4,7,8,9,10,11,12]]

    xt, yt = radar_patch(f1, theta, centre, scale_)
    xt1, yt1 = radar_patch(default_init, theta, centre, scale_)

    # Ismaila Sarr

    f1 = src_.loc[2][[4,7,8,9,10,11,12]]
    xt4, yt4 = radar_patch(f1, theta, centre, scale_)
    xt5, yt5 = radar_patch(default_init, theta, centre, scale_)

    # Adama Traore
    scale_ = scale.loc[2][2:]
    default_init = df_def.loc[2][2:]

    f1 = src_.loc[7][[4,7,8,9,10,11,12]]

    xt2, yt2 = radar_patch(f1, theta, centre, scale_)
    xt3, yt3 = radar_patch(default_init, theta, centre, scale_)

    # Raul Jimenez
    f1 = src_.loc[1][[4,7,8,9,10,11,12]]
    xt6, yt6 = radar_patch(f1, theta, centre, scale_)
    xt7, yt7 = radar_patch(default_init, theta, centre, scale_)

    # Houssem
    scale_ = scale.loc[5][2:]
    default_init = df_def.loc[5][2:]

    f1 = src_.loc[3][[4,7,8,9,10,11,12]]

    xt8, yt8 = radar_patch(f1, theta, centre, scale_)
    xt9, yt9 = radar_patch(default_init, theta, centre, scale_)
    # Victor

    scale_ = scale.loc[4][2:]
    default_init = df_def.loc[4][2:]

    f1 = src_.loc[4][[4,7,8,9,10,11,12]]

    xt10, yt10 = radar_patch(f1, theta, centre, scale_)
    xt11, yt11 = radar_patch(default_init, theta, centre, scale_)
    # Edison

    f1 = src_.loc[5][[4,7,8,9,10,11,12]]

    xt12, yt12 = radar_patch(f1, theta, centre, scale_)
    xt13, yt13 = radar_patch(default_init, theta, centre, scale_)

    # Kai
    scale_ = scale.loc[1][2:]
    default_init = df_def.loc[1][2:]

    f1 = src_.loc[8][[4,7,8,9,10,11,12]]

    xt14, yt14 = radar_patch(f1, theta, centre, scale_)
    xt15, yt15 = radar_patch(default_init, theta, centre, scale_)
    # David
    scale_ = scale.loc[0][2:]
    default_init = df_def.loc[0][2:]

    f1 = src_.loc[0][[4,7,8,9,10,11,12]]

    xt16, yt16 = radar_patch(f1, theta, centre, scale_)
    xt17, yt17 = radar_patch(default_init, theta, centre, scale_)

    # Ferran



    df = pd.DataFrame(data = {'x_cor' : [xt.values,xt1.values] ,'y_cor' : [yt.values, yt1.values]})
    df1 = pd.DataFrame(data = {'x_cor' : [xt2.values,xt3.values] ,'y_cor' : [yt2.values, yt3.values]})
    df2 = pd.DataFrame(data = {'x_cor' : [xt4.values,xt5.values] ,'y_cor' : [yt4.values, yt5.values]})
    df3 = pd.DataFrame(data = {'x_cor' : [xt6.values,xt7.values] ,'y_cor' : [yt6.values, yt7.values]})
    df4 = pd.DataFrame(data = {'x_cor' : [xt8.values,xt9.values] ,'y_cor' : [yt8.values, yt9.values]})
    df5 = pd.DataFrame(data = {'x_cor' : [xt10.values,xt11.values] ,'y_cor' : [yt10.values, yt11.values]})
    df6 = pd.DataFrame(data = {'x_cor' : [xt12.values,xt13.values] ,'y_cor' : [yt12.values, yt13.values]})
    df7 = pd.DataFrame(data = {'x_cor' : [xt14.values,xt15.values] ,'y_cor' : [yt14.values, yt15.values]})
    df8 = pd.DataFrame(data = {'x_cor' : [xt16.values,xt17.values] ,'y_cor' : [yt16.values, yt17.values]})

    patch_xy = (list(df['x_cor'].values), list(df['y_cor']))
    patch_xy1 = (list(df1['x_cor'].values), list(df1['y_cor']))
    patch_xy2 = (list(df2['x_cor'].values), list(df2['y_cor']))
    patch_xy3 = (list(df3['x_cor'].values), list(df3['y_cor']))
    patch_xy4 = (list(df4['x_cor'].values), list(df4['y_cor']))
    patch_xy5 = (list(df5['x_cor'].values), list(df5['y_cor']))
    patch_xy6 = (list(df6['x_cor'].values), list(df6['y_cor']))
    patch_xy7 = (list(df7['x_cor'].values), list(df7['y_cor']))
    patch_xy8 = (list(df8['x_cor'].values), list(df8['y_cor']))

    data_source = {
        'x':patch_xy[0],
        'y':patch_xy[1],
        'color': ['blue', 'purple'],
        'Jack Grealish_x' : patch_xy[0],
        'Jack Grealish_y' : patch_xy[1],
        'Adama Traoré_x' : patch_xy1[0],
        'Adama Traoré_y' : patch_xy1[1],
        'Ismaila Sarr_x' : patch_xy2[0],
        'Ismaila Sarr_y' : patch_xy2[1],
        'Raúl Jiménez_x' : patch_xy3[0],
        'Raúl Jiménez_y' : patch_xy3[1],
        'Houssem Aouar_x' : patch_xy4[0],
        'Houssem Aouar_y' : patch_xy4[1],
        'Victor Osimhen_x' : patch_xy5[0],
        'Victor Osimhen_y' : patch_xy5[1],
        'Edison Cavani_x' : patch_xy6[0],
        'Edison Cavani_y' : patch_xy6[1],
        'Kai Havertz_x' : patch_xy7[0],
        'Kai Havertz_y' : patch_xy7[1],
        'David Alaba_x' : patch_xy8[0],
        'David Alaba_y' : patch_xy8[1]
    }


    opts = {
    'Bundesliga' : ['Kai Havertz', 'David Alaba'],
    'Ligue 1' : ['Edison Cavani', 'Houssem Aouar', 'Victor Osimhen'],
    'EPL' : ['Jack Grealish', 'Ismaila Sarr', 'Adama Traoré', "Raúl Jiménez"],
    "Seria A" : ['Gianluigi Donnaruma']
    }

    source = ColumnDataSource(data_source)

    #########

    file = 'Draft.csv'
    tweet_data = pd.read_csv(file)
    tweet_data = tweet_data[['created_at',
       'player', 'sentiment', 'country']]

    tweet_data = tweet_data[tweet_data['player'] != 'Ferran']

    from datetime import datetime, date, timedelta

    fmt = '%B-%d'
    tweet_data.created_at = pd.to_datetime(tweet_data['created_at']).apply(lambda x: x.date())

    color_dic = { '1' : RdBu3[0], '0': RdBu3[2]} 

    from math import ceil
    def tweet_date_count(df, player, dic):
        tweets = df[(df['created_at'] >= date(2020,7,25)) & (df['player'] == player)]
        tweets = tweets.set_index(pd.DatetimeIndex(tweets['created_at'])).drop('created_at', axis= 1).sort_index()
        if tweets.index[0] != date(2020,7,25):
            tweets.loc[date(2020,7,25)]={'country' : np.nan, 'player': player, 'sentiment':0} 
        tweets.index = pd.to_datetime(tweets.index)
        tweets.sort_index(inplace = True)
        if tweets.index[-1] != date(2020,8,12):
            tweets.loc[date(2020,8,12)]=[player, np.nan, np.nan] 
        tweets.index = pd.to_datetime(tweets.index)
        tweets.sort_index(inplace = True)
        tweets_sum = tweets['sentiment'].resample('3D').sum()
        tweets_count = tweets['sentiment'].resample('3D').count()
        tweets_count = tweets_count.replace(0,np.nan)
        tweets_proportion = tweets_sum / tweets_count
        tweets_proportion = tweets_proportion.fillna(0)
    
    
        xr = tweets_proportion.sort_index(ascending = True).index
        xr = [datetime.strftime(i, fmt) for i in xr]
        yr = tweets_proportion.sort_index(ascending = True).values
        pal = [dic[str(ceil(y))] for y in yr]
        return xr, yr, pal

    values = []    
    players_ = ['Jack', 'Adama', 'Ismaila', 'Raul', 'Houssem', 'Victor', 'Edison', 'Kai', 'David']
    ##
    # Jack Grealish

    for player in players_:
        values.append(tweet_date_count(tweet_data,player, color_dic))

    data_src = {
        'x':values[0][0],
        'y':values[0][1],
        'color': values [0][2],
        'Jack Grealish_y' : values[0][1],
        'Jack Grealish_c' : values[0][2],
        'Adama Traoré_y' : values[1][1],
        'Adama Traoré_c' : values[1][2],
        'Ismaila Sarr_y' : values[2][1],
        'Ismaila Sarr_c' : values[2][2],
        'Raúl Jiménez_y' : values[3][1],
        'Raúl Jiménez_c' : values[3][2],
        'Houssem Aouar_y' : values[4][1],
        'Houssem Aouar_c' : values[4][2],
        'Victor Osimhen_y' : values[5][1],
        'Victor Osimhen_c' : values[5][2],
        'Edison Cavani_y' : values[6][1],
        'Edison Cavani_c' : values[6][2],
        'Kai Havertz_y' : values[7][1],
        'Kai Havertz_c' : values[7][2],
        'David Alaba_y' : values[8][1],
        'David Alaba_c' : values[8][2]
    }

    ##Ferran

    data_src = ColumnDataSource(data_src)


    import geopandas as gpd

    gdf = gpd.read_file('https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson')
    gdf = gdf[gdf.NAME != 'Antarctica']

    ### Boundaries
    xs = []
    ys = []
    for obj in gdf.geometry.boundary:
        if obj.type == 'LineString':
            obj_x, obj_y = obj.xy
            xs.append([[list(obj_x)]])
            ys.append([[list(obj_y)]])
        elif obj.type == 'MultiLineString':
            obj_x = []
            obj_y = []
            for line in obj:
                line_x, line_y = line.xy
                obj_x.append([list(line_x)])
                obj_y.append([list(line_y)])
            xs.append(obj_x)
            ys.append(obj_y)

    country = gdf['NAME'].values        

    df3_plot = pd.DataFrame({'country': country, 'xs': xs, 'ys': ys})


    tweet_data.dropna(axis = 0, inplace = True)
    countries = tweet_data.country
    to_replace = countries[~countries.isin(df3_plot.country)].unique()

    replc = {'United States' : 'United States of America', 'Myanmar (Burma)': 'Myanmar', 'Bosnia and Herzegovina':'Bosnia and Herz.', 'Eswatini': 'Swaziland'}

    tweet_data['country'] = tweet_data['country'].replace(replc)

    map4 = tweet_data[['player', 'country']]

    map4 = pd.get_dummies(map4, columns = ['country'], prefix = '', prefix_sep = '')
    map4 = map4.groupby('player', as_index = False).sum()

    map4 = map4.set_index('player').reindex(players_).reset_index()

    country = df3_plot.country
    diff = country[~country.isin(map4.columns)].values
    map4 = pd.concat([map4, pd.DataFrame(columns = diff)], axis = 1).fillna(0)
    map4 = map4.loc[:, np.append(['player'], country.values)]

    ##Jack
    dfs_plot = pd.DataFrame()

    for i, j in enumerate(players_):
        temp = df3_plot.copy()
        temp = pd.merge(temp, pd.DataFrame(map4.iloc[i,1:].reset_index()), left_on = 'country', right_on = 'index').drop('index', axis = 1)
        temp.columns = ['country', 'xs','ys','count']
        temp['player'] = j
        dfs_plot = dfs_plot.append(temp)

    data_plot = {
        'xs':dfs_plot[dfs_plot['player'] == 'Jack']['xs'].values,
        'ys':dfs_plot[dfs_plot['player'] == 'Jack']['ys'].values,
        'country' : dfs_plot[dfs_plot['player'] == 'Jack']['country'].values,
        'count' : dfs_plot[dfs_plot['player'] == 'Jack']['count'].values,
        'Jack Grealish' : dfs_plot[dfs_plot['player'] == 'Jack']['count'].values,
        'Adama Traoré' : dfs_plot[dfs_plot['player'] == 'Adama']['count'].values,
        'Ismaila Sarr' : dfs_plot[dfs_plot['player'] == 'Ismaila']['count'].values,
        'Raúl Jiménez' : dfs_plot[dfs_plot['player'] == 'Raul']['count'].values,
        'Houssem Aouar' : dfs_plot[dfs_plot['player'] == 'Houssem']['count'].values,
        'Victor Osimhen' : dfs_plot[dfs_plot['player'] == 'Victor']['count'].values,
        'Edison Cavani' : dfs_plot[dfs_plot['player'] == 'Edison']['count'].values,
        'Kai Havertz' : dfs_plot[dfs_plot['player'] == 'Kai']['count'].values,
        'David Alaba' : dfs_plot[dfs_plot['player'] == 'David']['count'].values
    }

    s3 = ColumnDataSource(data_plot)

    ### xG, xA Delta

    delta = data.loc[(transfers, '2019/2020') , ['deltaXG', 'deltaXA']].iloc[[0,1,2,3,4,5,6,8,11]].reset_index().drop('Season', axis = 1)

    line_colo = []
    for i in delta.index:
        colo = ['red' if a >= 0 else 'green' for a in delta.iloc[i,1:].values]
        line_colo.append(colo)

    src_delta = {
        'factors' : ["xG - G", "xA - A"],
        'x0' : [0,0],
        'x': list(delta.iloc[6,1:].values),
        'colo' : line_colo[6],
        'size' : [15,15],
        'Jack Grealish' : list(delta.iloc[6,1:].values),
        'Jack Grealish_x' : line_colo[6],
        'Adama Traoré' : list(delta.iloc[7,1:].values),
        'Adama Traoré_x' : line_colo[7],
        'Ismaila Sarr' : list(delta.iloc[2,1:].values),
        'Ismaila Sarr_x' : line_colo[2],
        'Raúl Jiménez' : list(delta.iloc[1,1:].values),
        'Raúl Jiménez_x' : line_colo[1],
        'Houssem Aouar' : list(delta.iloc[3,1:].values),
        'Houssem Aouar_x' : line_colo[3],
        'Victor Osimhen' : list(delta.iloc[4,1:].values),
        'Victor Osimhen_x' : line_colo[4],
        'Edison Cavani' :list(delta.iloc[5,1:].values),
        'Edison Cavani_x': line_colo[5],
        'Kai Havertz' : list(delta.iloc[8,1:].values),
        'Kai Havertz_x' : line_colo[8],
        'David Alaba' : list(delta.iloc[0,1:].values),
        'David Alaba_x' : line_colo[0]
    }

    src_del = ColumnDataSource(src_delta)


    dot = figure(plot_width=720, plot_height=220, title = 'BALANCE DE GOLES & ASISTENCIAS', toolbar_location = None, x_range = [-2.2,3], y_range = src_delta['factors'])

    dot.segment('x0', 'factors', 'x', 'factors', line_color = 'colo', line_width = 4, source = src_del)

    dot.circle(x = 'x', y =  'factors', size = 'size', fill_color = 'orange', line_width = 3, line_color = 'colo', source = src_del)

    ###p1
    rendered = p.patches('x', 'y', source = source, fill_color='color', fill_alpha=0.45)

    xdr = FactorRange(factors=data_src.data['x'])
    ydr = Range1d(start = -.3, end = 1)

    p1 = figure(plot_width=600, plot_height=340, title = 'SENTIMENT ANALYSIS: TWEETS DEL JUGADOR EN TEMPORADA DE TRANSFERENCIAS', x_range=xdr, y_range = ydr, 
            x_axis_label='FECHA', y_axis_label='SENTIMENT SCORE', toolbar_location=None)

    ###p2
    rendered1 = p1.vbar(x='x', top='y', width=.7,source=data_src, fill_alpha = .6, fill_color = 'color')
    ##option = p1.line('index', 'defects', line_width=2, color='red', source=fill_source)

    ###p3
    colorsrev = tuple(reversed(Blues8))
    cmap = linear_cmap('count', palette=colorsrev, low=0, high= 10)

    # Create the Figure object "p2"

    p2 = figure(plot_width= 810, plot_height=400, title = 'SENTIMENT ANALYSIS: TWEETS DEL JUGADOR POR PAIS',
            toolbar_location=None, tools=['hover', 'pan'], tooltips='@country: @count')

    p2.multi_polygons(xs='xs', ys='ys', fill_color=cmap , source=s3)



    callback = CustomJS(args=dict(source=source,plot=p, rendered = rendered, rendered1 = rendered1, src = data_src,
                    xrange=p1.x_range, p2=p2, s_data = s3, l_src = lines_src, delta_src = src_del), code="""  
        var comp = cb_obj.value;
        var data = source.data;
        var keyx = String(comp)+'_x';
        var keyy = String(comp)+'_y';
        var keyc = String(comp)+'_c';

        var data0 = l_src.data;
        var keyl = String(comp)+'_l';
        var keyl1 = String(comp)+'_l1';
        var keyl2 = String(comp)+'_l2';
        data0['text1'] = data0[String(keyl)]
        data0['text2'] = data0[String(keyl1)]
        data0['text3'] = data0[String(keyl2)]
        l_src.change.emit();

        data['x'][0] = data[String(keyx)][0]
        data['x'][1] =data[String(keyx)][1]
        data['y'][0] = data[String(keyy)][0]
        data['y'][1] = data[String(keyy)][1]
        source.change.emit();

        var data1 = src.data;
        var keyy1 = String(comp) + '_y';
        data1['y'] = data1[String(keyy1)]
        data1['color'] = data1[String(keyc)]
        src.change.emit();  
        xrange.factors = data1['x'];
        src.change.emit();

        var data2 = s_data.data;
        var country = data2['country'];
        data2['count'] = data2[comp]
        s_data.change.emit();

        var datax = delta_src.data;
        var keycolo = String(comp)+'_x';
        datax['colo'] = datax[String(keycolo)]
        datax['x'] = datax[comp]
        delta_src.change.emit();
    """ 
    )

    callback2 = CustomJS(args=dict(p3=p2), code='''
        p3.reset.emit();
    ''')

    select1 = Select(title = 'League', options = ['EPL', 'Bundesliga','Ligue 1', 'Serie A'], value = 'EPL')

    select2 = Select(title = 'Player', options = transfers, value = 'Jack Grealish')

    select2.js_on_change('value', callback)

    select1.js_on_change('value', CustomJS(args=dict(select2=select2), code="""
        const opts = %s
        select2.options = opts[cb_obj.value]
    """ % opts))

    button = Button(label='Reset map view')
    button.js_on_click(callback2)

    p.axis.visible = False
    p.grid.grid_line_color = None
    p1.grid.grid_line_color = None

    p2.axis.visible = False
    p2.grid.grid_line_color = None

    widget = row(select1,select2)

    layout = row(column(p1, p2))
    
    st.title('Fútbol de Europa: Evaluando Pases de Verano (Posibles y Completados)')
    st.header('Jugadores vs el promedio')
    st.markdown(' El promedio se basa en la posición del jugador en su respectiva liga. Como')
    st.bokeh_chart(row(column(widget, p),dot))  
    st.header('Análisis de Sentiment basado en Tweets acerca del Jugador')
    st.boke_chart(layout)


if __name__ == '__main__':
    main()

