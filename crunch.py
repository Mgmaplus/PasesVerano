import streamlit as st
from requests_html import HTMLSession
import pandas as pd
from os import listdir
from os.path import join, isfile

def main():
    ## main streamlit app to host on heroku
    
    dfs = ['defenders_bundesliga.csv', 'forwards_bundesliga.csv', 
           'forwards_epl.csv', 'forwards_ligue1.csv', 'midfielders_epl.csv', 
           'midfielders_ligue1.csv', 'midfielders_laliga.csv', 'forwards_laliga.csv']
    
    data = pd.DataFrame(columns = ['№', 'Season', 'Team', 'Apps', 'Min', 'G', 'A', 'Sh90', 'KP90', 'xG',
       'xA', 'xG90', 'xA90', 'Player', 'League', 'Pos'])
    for file in dfs:
        temp = pd.read_csv(file)
        temp.drop('Unnamed: 0', axis = 1, inplace = True)
        data = data.append(temp)

    files = ['porteros_serieA.csv', 'porteros1_serieA.csv']
    serie_a = pd.read_csv(files[0])
    merge = pd.read_csv(files[1])
    serie_a = serie_a.merge(merge, on = 'Player')
    serie_a.drop(['Rk_x', 'Rk_y','Nation_x','Nation_y', 'Pos_x', 'Pos_y', 'Squad_x', 'Squad_y', 'Age_x', 'Age_y', 'Born_x', 'Born_y', 'GA_x','GA_y', 'PKA_x', 'PKA_y', 'Matches_x', 'Matches_y',],
                axis = 1, inplace = True )
    serie_a['Player'] = serie_a['Player'].apply(lambda x: x.split('\\')[0]) 

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
    data['Min'] = data['Min'] / 90

    data[['Apps', 'G', 'A']] = data[['Apps', 'G', 'A']].astype(int)
    
    data.loc[('Lionel Messi', '2018/2019'), 'xG'] = '26.0'
    data.loc[('Lionel Messi', '2018/2019'),'deltaXG'] = '-10.0' 
    data.loc[('Lionel Messi', '2016/2017'), 'xG'] = '26.89' 
    data.loc[('Lionel Messi', '2016/2017'),'deltaXG'] = '-10.11' 

    data[['Min','xG','xA','deltaXG', 'deltaXA']] = data[['Min','xG','xA','deltaXG', 'deltaXA']].astype(float)
    

    transfers = ['David Alaba', 'Raúl Jiménez','Ismaila Sarr', 'Houssem Aouar', 'Victor Osimhen', 
            'Edinson Cavani', 'Jack Grealish', 'Adama Traoré','Kai Havertz', 'Ferrán Torres', 'Lionel Messi']

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
    from bokeh.models.widgets import Paragraph, Div

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
    scale1 = data[data['Min'] >=5.55].groupby(['League','Pos']).max()[['Min','Sh90','KP90','xG','xA','xG90','xA90']].reset_index()

    #Gianluigi
    serie_a['Penales_per'] = serie_a['PKsv'] / serie_a['PKatt'] *100
    scale_por = serie_a[serie_a['90s'] >= 5.55].max()[['90s','PSxG','FK','AvgLen','GA90','Penales_per', 'CS%']]

    scale_75 = scale_por * .75
    labels75_g = [f"{num:.1f}" for num in scale_75] 
    item1 = labels75_g[0]
    labels75_g.reverse()
    labels75_g = [item1] + labels75_g[0:6] + ['']
    scale_50 = scale_por * .50
    labels50_g = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_g[0]
    labels50_g.reverse()
    labels50_g = [item1] + labels50_g[0:6] + ['']
    scale_25 = scale_por * .25
    labels25_g = [f"{num:.1f}" for num in scale_25]
    item1 = labels25_g[0]
    labels25_g.reverse()
    labels25_g = [item1] + labels25_g[0:6] + ['']



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
    scale_75 = scale1.loc[7][2:] *.75
    labels75_3 = [f"{num:.1f}" for num in scale_75] 
    item1 = labels75_3[0]
    labels75_3.reverse()
    labels75_3 = [item1] + labels75_3[0:6] + ['']
    scale_50 = scale1.loc[7][2:] *.5
    labels50_3 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_3[0]
    labels50_3.reverse()
    labels50_3 = [item1] + labels50_3[0:6] + ['']
    scale_25 = scale1.loc[7][2:] *.25
    labels25_3 = [f"{num:.1f}" for num in scale_25] 
    item1 = labels25_3[0]
    labels25_3.reverse()
    labels25_3 = [item1] + labels25_3[0:6] + ['']


    # Victor  # Edison

    scale_75 = scale1.loc[6][2:] *.75
    labels75_4 = [f"{num:.1f}" for num in scale_75]
    item1 = labels75_4[0]
    labels75_4.reverse()
    labels75_4 = [item1] + labels75_4[0:6] + ['']
    scale_50 = scale1.loc[6][2:] *.5
    labels50_4 = [f"{num:.1f}" for num in scale_50] 
    item1 = labels50_4[0]
    labels50_4.reverse()
    labels50_4 = [item1] + labels50_4[0:6] + ['']
    scale_25 = scale1.loc[6][2:] *.25
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
    scale_75 = scale1.loc[5][2:] *.75
    labels75_7 = [f"{num:.2f}" for num in scale_75]
    item1 = labels75_7[0]
    labels75_7.reverse()
    labels75_7 = [item1] + labels75_7[0:6] + ['']
    scale_50 = scale1.loc[5][2:] *.5
    labels50_7 = [f"{num:.1f}" for num in scale_50]
    item1 = labels50_7[0]
    labels50_7.reverse()
    labels50_7 = [item1] + labels50_7[0:6] + ['']
    scale_25 = scale1.loc[5][2:] *.25
    labels25_7 = [f"{num:.1f}" for num in scale_25]
    item1 = labels25_7[0]
    labels25_7.reverse()
    labels25_7 = [item1] + labels25_7[0:6] + ['']
    # Messi
    scale_75 = scale1.loc[4][2:] *.75
    labels75_8 = [f"{num:.1f}" for num in scale_75]
    item1 = labels75_8[0]
    labels75_8.reverse()
    labels75_8 = [item1] + labels75_8[0:6] + ['']
    scale_50 = scale1.loc[4][2:] *.5
    labels50_8 = [f"{num:.1f}" for num in scale_50]
    item1 = labels50_8[0]
    labels50_8.reverse()
    labels50_8 = [item1] + labels50_8[0:6] + ['']
    scale_25 = scale1.loc[4][2:] *.25
    labels25_8 = [f"{num:.1f}" for num in scale_25]
    item1 = labels25_8[0]
    labels25_8.reverse()
    labels25_8 = [item1] + labels25_8[0:6] + ['']

    ## x and y
    df_lab = pd.DataFrame(data = {'x_cor' : [x, x75, x50, x25] ,'y_cor' : [ y, y75, y50, y25]})
    text_g = ['Min', 'Inbatida(%)','PK(%)', 'Goles90', 'Saques(mts)', 'FK','PSxG','']

    ## text labels
    df1_lab = pd.DataFrame(data = {'text' : [text, labels75_1,labels50_1,labels25_1]})
    df2_lab = pd.DataFrame(data = {'text' : [text, labels75_2,labels50_2,labels25_2]})
    df3_lab = pd.DataFrame(data = {'text' : [text, labels75_3,labels50_3,labels25_3]})
    df4_lab = pd.DataFrame(data = {'text' : [text, labels75_4,labels50_4,labels25_4]})
    df5_lab = pd.DataFrame(data = {'text' : [text, labels75_5,labels50_5,labels25_5]})
    df6_lab = pd.DataFrame(data = {'text' : [text, labels75_6,labels50_6,labels25_6]})
    df7_lab = pd.DataFrame(data = {'text' : [text, labels75_7,labels50_7,labels25_7]})
    df8_lab = pd.DataFrame(data = {'text' : [text, labels75_8,labels50_8,labels25_8]})
    dfg_lab = pd.DataFrame(data = {'text' : [text_g, labels75_g, labels50_g, labels25_g]})



    label_xy = (list(df_lab['x_cor'].values), list(df_lab['y_cor']))
    label_1 = list(df1_lab['text'].values)
    label_2 = list(df2_lab['text'].values)
    label_3 = list(df3_lab['text'].values)
    label_4 = list(df4_lab['text'].values)
    label_5 = list(df5_lab['text'].values)
    label_6 = list(df6_lab['text'].values)
    label_7 = list(df7_lab['text'].values)
    label_8 = list(df8_lab['text'].values)
    label_g = list(dfg_lab['text'].values)

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
        'Edinson Cavani_l' : label_4[1],
        'Edinson Cavani_l1' : label_4[2],
        'Edinson Cavani_l2' : label_4[3],
        'Kai Havertz_l' : label_5[1],
        'Kai Havertz_l1' : label_5[2],
        'Kai Havertz_l2' : label_5[3],
        'David Alaba_l' : label_6[1],
        'David Alaba_l1' : label_6[2],
        'David Alaba_l2' : label_6[3],
        'text_g' : label_g[0],
        'Gianluigi Donnarumma_l' : label_g[1],
        'Gianluigi Donnarumma_l1' : label_g[2],
        'Gianluigi Donnarumma_l2' : label_g[3],
        'Ferrán Torres_l' : label_7[1],
        'Ferrán Torres_l1' : label_7[2],
        'Ferrán Torres_l2' : label_7[3],
        'Lionel Messi_l' : label_8[1],
        'Lionel Messi_l1' : label_8[2],
        'Lionel Messi_l2' : label_8[3],
        'text_def' : label_1[0] 
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
    scale = data[data['Min'] >=5.5].groupby(['League','Pos']).max()[['Min','Sh90','KP90','xG','xA','xG90','xA90']].reset_index()
    df_def = data[data['Min'] >=5.5].groupby(['League','Pos']).mean()[['Min','Sh90','KP90','xG','xA','xG90','xA90']].reset_index()   
    src_ = data.loc[transfers,'2019/2020',:].iloc[[0,1,2,3,4,5,6,8,11,12,13]].reset_index()
    

    # Gianluigi
    
    prom_g = serie_a[serie_a['90s'] >= 5.5].mean()[['90s','PSxG','FK','AvgLen','GA90','Penales_per', 'CS%']]
    gianluigi = serie_a[serie_a['Player'] == 'Gianluigi Donnarumma'][['90s','PSxG','FK','AvgLen','GA90','Penales_per', 'CS%']]
    
    xtg, ytg = radar_patch(gianluigi, theta, centre, scale_por)
    xtg1, ytg1 = radar_patch(prom_g, theta, centre, scale_por)

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
    scale_ = scale.loc[7][2:]
    default_init = df_def.loc[7][2:]

    f1 = src_.loc[3][[4,7,8,9,10,11,12]]

    xt8, yt8 = radar_patch(f1, theta, centre, scale_)
    xt9, yt9 = radar_patch(default_init, theta, centre, scale_)
    # Victor

    scale_ = scale.loc[6][2:]
    default_init = df_def.loc[6][2:]

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
    scale_ = scale.loc[5][2:]
    default_init = df_def.loc[5][2:]

    f1 = src_.loc[9][[4,7,8,9,10,11,12]]

    xt18, yt18 = radar_patch(f1, theta, centre, scale_)
    xt19, yt19 = radar_patch(default_init, theta, centre, scale_)

    # Messi
    scale_ = scale.loc[4][2:]
    default_init = df_def.loc[4][2:]

    f1 = src_.loc[10][[4,7,8,9,10,11,12]]

    xt20, yt20 = radar_patch(f1, theta, centre, scale_)
    xt21, yt21 = radar_patch(default_init, theta, centre, scale_)


    df = pd.DataFrame(data = {'x_cor' : [xt.values,xt1.values] ,'y_cor' : [yt.values, yt1.values]})
    df1 = pd.DataFrame(data = {'x_cor' : [xt2.values,xt3.values] ,'y_cor' : [yt2.values, yt3.values]})
    df2 = pd.DataFrame(data = {'x_cor' : [xt4.values,xt5.values] ,'y_cor' : [yt4.values, yt5.values]})
    df3 = pd.DataFrame(data = {'x_cor' : [xt6.values,xt7.values] ,'y_cor' : [yt6.values, yt7.values]})
    df4 = pd.DataFrame(data = {'x_cor' : [xt8.values,xt9.values] ,'y_cor' : [yt8.values, yt9.values]})
    df5 = pd.DataFrame(data = {'x_cor' : [xt10.values,xt11.values] ,'y_cor' : [yt10.values, yt11.values]})
    df6 = pd.DataFrame(data = {'x_cor' : [xt12.values,xt13.values] ,'y_cor' : [yt12.values, yt13.values]})
    df7 = pd.DataFrame(data = {'x_cor' : [xt14.values,xt15.values] ,'y_cor' : [yt14.values, yt15.values]})
    df8 = pd.DataFrame(data = {'x_cor' : [xt16.values,xt17.values] ,'y_cor' : [yt16.values, yt17.values]})
    df9 = pd.DataFrame(data = {'x_cor' : [xt18.values,xt19.values] ,'y_cor' : [yt18.values, yt19.values]})
    df10 = pd.DataFrame(data = {'x_cor' : [xt20.values,xt21.values] ,'y_cor' : [yt20.values, yt21.values]})
    dfg = pd.DataFrame(data = {'x_cor' : [xtg.values[0], xtg1.values], 'y_cor' : [ytg.values[0], ytg1.values]})

    patch_xy = (list(df['x_cor'].values), list(df['y_cor']))
    patch_xy1 = (list(df1['x_cor'].values), list(df1['y_cor']))
    patch_xy2 = (list(df2['x_cor'].values), list(df2['y_cor']))
    patch_xy3 = (list(df3['x_cor'].values), list(df3['y_cor']))
    patch_xy4 = (list(df4['x_cor'].values), list(df4['y_cor']))
    patch_xy5 = (list(df5['x_cor'].values), list(df5['y_cor']))
    patch_xy6 = (list(df6['x_cor'].values), list(df6['y_cor']))
    patch_xy7 = (list(df7['x_cor'].values), list(df7['y_cor']))
    patch_xy8 = (list(df8['x_cor'].values), list(df8['y_cor']))
    patch_xy9 = (list(df9['x_cor'].values), list(df9['y_cor']))
    patch_xy10 = (list(df10['x_cor'].values), list(df10['y_cor']))
    patch_g = (list(dfg['x_cor']), list(dfg['y_cor']))

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
        'Edinson Cavani_x' : patch_xy6[0],
        'Edinson Cavani_y' : patch_xy6[1],
        'Kai Havertz_x' : patch_xy7[0],
        'Kai Havertz_y' : patch_xy7[1],
        'David Alaba_x' : patch_xy8[0],
        'David Alaba_y' : patch_xy8[1],
        'Gianluigi Donnarumma_x' : patch_g[0],
        'Gianluigi Donnarumma_y' : patch_g[1],
        'Ferrán Torres_x' : patch_xy9[0],
        'Ferrán Torres_y' : patch_xy9[1],
        'Lionel Messi_x' : patch_xy10[0],
        'Lionel Messi_y' : patch_xy10[1]
    }


    opts = {
    'Bundesliga' : ['Seleccionar...','Kai Havertz', 'David Alaba'],
    'Ligue 1' : ['Seleccionar...','Edinson Cavani', 'Houssem Aouar', 'Victor Osimhen'],
    'La Liga' : ['Seleccionar...', 'Ferrán Torres', 'Lionel Messi'],
    'Premier League' : ['Seleccionar...','Jack Grealish', 'Ismaila Sarr', 'Adama Traoré', "Raúl Jiménez"],
    'Serie A' : ['Seleccionar...','Gianluigi Donnarumma']
    }

    source = ColumnDataSource(data_source)

    #########

    file = 'FinalTweets.csv'
    tweet_data = pd.read_csv(file)
    tweet_data = tweet_data[['created_at',
       'player', 'sentiment', 'country']]

    ##tweet_data = tweet_data[tweet_data['player'] != 'Ferran']

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

    def tweet_date_count_b(df, player, dic):
        tweets = df[df['player'] == player]
        tweets = tweets.set_index(pd.DatetimeIndex(tweets['created_at'])).drop('created_at', axis= 1).sort_index()
        tweets_sum = tweets['sentiment'].resample('1D').sum()
        tweets_count = tweets['sentiment'].resample('1D').count()
        tweets_count = tweets_count.replace(0,np.nan)
        tweets_proportion = tweets_sum / tweets_count
        tweets_proportion = tweets_proportion.fillna(0)
    
        xr = tweets_proportion.sort_index(ascending = True).index
        xr = [datetime.strftime(i, fmt) for i in xr]
        yr = tweets_proportion.sort_index(ascending = True).values
        pal = [dic[str(ceil(y))] for y in yr]
        if player == 'Lionel':
            return xr[1:-1], yr[1:-1], pal[1:-1]
        if player == 'Ferran':
            pre = [0,0]
            yr = list(yr)
            yr.append(0)
            for i in yr:
             pre.append(i)
            pre.append(0)
            return ['July-12', 'July-13'] + xr + ['July-17', 'July-18'], np.array(pre) , ['#67a9cf','#67a9cf'] + pal + ['#67a9cf','#67a9cf']

    values = []    
    players_ = ['Jack', 'Adama', 'Ismaila', 'Raul', 'Houssem', 'Victor', 'Edison', 'Kai', 'David', 'Gianluigi']
    players_b = ['Ferran', 'Lionel']
    ##
    # Jack Grealish

    for player in players_:
        values.append(tweet_date_count(tweet_data,player, color_dic))
    
    for player in players_b:
        values.append(tweet_date_count_b(tweet_data, player, color_dic))

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
        'Edinson Cavani_y' : values[6][1],
        'Edinson Cavani_c' : values[6][2],
        'Kai Havertz_y' : values[7][1],
        'Kai Havertz_c' : values[7][2],
        'David Alaba_y' : values[8][1],
        'David Alaba_c' : values[8][2],
        'Gianluigi Donnarumma_y' : values[9][1],
        'Gianluigi Donnarumma_c' : values[9][2],
        'Ferrán Torres_y' : values[10][1],
        'Ferrán Torres_c' : values[10][2],
        'Lionel Messi_y' : values[11][1],
        'Lionel Messi_c' : values[11][2],
        'Ferrán Torres' : values[10][0],
        'Lionel Messi' : values[11][0],
        'default' : values[0][0]
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
    all_play = players_ + players_b
    map4 = map4.set_index('player').reindex(all_play).reset_index()

    country = df3_plot.country
    diff = country[~country.isin(map4.columns)].values
    map4 = pd.concat([map4, pd.DataFrame(columns = diff)], axis = 1).fillna(0)
    map4 = map4.loc[:, np.append(['player'], country.values)]

    ##Jack
    dfs_plot = pd.DataFrame()
    
    for i, j in enumerate(all_play):
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
        'Edinson Cavani' : dfs_plot[dfs_plot['player'] == 'Edison']['count'].values,
        'Kai Havertz' : dfs_plot[dfs_plot['player'] == 'Kai']['count'].values,
        'David Alaba' : dfs_plot[dfs_plot['player'] == 'David']['count'].values,
        'Gianluigi Donnarumma' : dfs_plot[dfs_plot['player'] =='Gianluigi']['count'].values,
        'Ferrán Torres' : dfs_plot[dfs_plot['player'] =='Ferran']['count'].values,
        'Lionel Messi' : dfs_plot[dfs_plot['player'] =='Lionel']['count'].values
    }

    s3 = ColumnDataSource(data_plot)

    ### xG, xA Delta

    delta = data.loc[(transfers, '2019/2020') , ['deltaXG', 'deltaXA']].iloc[[0,1,2,3,4,5,6,8,11,12,13]].reset_index().drop('Season', axis = 1)

    #Gianluigi 
    delta_g = serie_a[serie_a['Player'] == 'Gianluigi Donnarumma'][['PSxG+/-', '/90']] * -1

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
        'Edinson Cavani' :list(delta.iloc[5,1:].values),
        'Edinson Cavani_x': line_colo[5],
        'Kai Havertz' : list(delta.iloc[8,1:].values),
        'Kai Havertz_x' : line_colo[8],
        'David Alaba' : list(delta.iloc[0,1:].values),
        'David Alaba_x' : line_colo[0],
        'Ferrán Torres' : list(delta.iloc[9,1:].values),
        'Ferrán Torres_x' : line_colo[9],
        'Lionel Messi' : list(delta.iloc[10,1:].values),
        'Lionel Messi_x' : line_colo[10],
        'Gianluigi Donnarumma' : [delta_g.loc[11,'PSxG+/-'], delta_g.loc[11,'/90']],
        'Gianluigi Donnarumma_x' : ['green', 'green'],
        'factors_g' : ["Goles - PSxG", "G90 - PSxG90"],
        'factors_def' : ["xG - G", "xA - A"]
    }

    src_del = ColumnDataSource(src_delta)

    dot = figure(plot_width=720, plot_height=220, title = 'BALANCE DE VALORES ESPERADOS', toolbar_location = None, x_range = [-5,5.5], y_range = src_del.data['factors'])

    dot.segment('x0', 'factors', 'x', 'factors', line_color = 'colo', line_width = 4, source = src_del)

    dot.circle(x = 'x', y =  'factors', size = 'size', fill_color = 'orange', line_width = 3, line_color = 'colo', source = src_del)


    ###p1
    rendered = p.patches('x', 'y', source = source, fill_color='color', fill_alpha=0.45)

    xdr = FactorRange(factors=data_src.data['x'])
    ydr = Range1d(start = -.3, end = 1)

    p1 = figure(plot_width=630, plot_height=340, title = 'TWEETS SOBRE EL JUGADOR: SENTIMENT (VALOR) DURANTE MERCADO DE PASES', x_range=xdr, y_range = ydr, 
            x_axis_label='FECHA', y_axis_label='SENTIMENT SCORE', toolbar_location=None)

    ###p2
    rendered1 = p1.vbar(x='x', top='y', width=.7,source=data_src, fill_alpha = .6, fill_color = 'color')

    ###p3
    colorsrev = tuple(reversed(Blues8))
    cmap = linear_cmap('count', palette=colorsrev, low=0, high= 10)

    # Create the Figure object "p2"

    p2 = figure(plot_width= 810, plot_height=400, title = 'LOS TWEETS SOBRE EL JUGADOR POR PAIS',
            toolbar_location=None, tools=['hover', 'pan'], tooltips='@country: @count')

    p2.multi_polygons(xs='xs', ys='ys', fill_color=cmap , source=s3)



    callback = CustomJS(args=dict(source=source,plot=p, rendered = rendered, rendered1 = rendered1, src = data_src,
                    xrange=p1.x_range, p2=p2, s_data = s3, l_src = lines_src, delta_src = src_del, factors = dot.y_range), code="""  
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
        if (String(comp) == 'Gianluigi Donnarumma'){
            data0['text'] = data0['text_g']}
        else{
            data0['text'] = data0['text_def']}
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
        if (String(comp) == 'Lionel Messi'){
            xrange.factors = data1['Lionel Messi']
            data1['x'] = data1['Lionel Messi']
        }
        else {
            if (String(comp)== 'Ferrán Torres'){
            xrange.factors = data1['Ferrán Torres']
            data1['x'] = data1['Ferrán Torres']
        }
           else{
            xrange.factors = data1['default']
            data1['x'] = data1['default']}
        }
        src.change.emit();

        var data2 = s_data.data;
        var country = data2['country'];
        data2['count'] = data2[comp]
        s_data.change.emit();

        var datax = delta_src.data;
        var keycolo = String(comp)+'_x';
        datax['colo'] = datax[String(keycolo)]
        datax['x'] = datax[comp]
        if (String(comp) == 'Gianluigi Donnarumma'){
            factors.factors = datax['factors_g']
            datax['factors'] = datax['factors_g']
        }
        else {
            factors.factors = datax['factors_def']
            datax['factors'] = datax['factors_def']
        }
        delta_src.change.emit();
    """ 
    )

    players_all = ['Seleccionar...'] + transfers + ['Gianluigi Donnarumma']
    select1 = Select(title = 'League', options = ['Bundesliga', 'La Liga', 'Premier League', 'Ligue 1', 'Serie A'], value = 'Premier League')

    select2 = Select(title = 'Player', options = players_all, value = 'Seleccionar...')

    select1.js_on_change('value', CustomJS(args=dict(select2=select2), code="""
        const opts = %s
        select2.options = opts[cb_obj.value]
    """ % opts))

    select2.js_on_change('value', callback)

   

    p.axis.visible = False
    p.grid.grid_line_color = None
    p1.grid.grid_line_color = None

    p2.axis.visible = False
    p2.grid.grid_line_color = None

    p.toolbar.active_drag = None
    p.toolbar.active_scroll = None

    dot.toolbar.active_drag = None
    dot.toolbar.active_scroll = None

    p1.toolbar.active_drag = None
    p1.toolbar.active_scroll = None

    p2.toolbar.active_drag = None
    p2.toolbar.active_scroll = None

    widget = row(select1,select2)
    
    st.title('Fútbol de Europa: Evaluando los Posibles y Completados Pases de Verano 2020')
    st.header('Contrastando con el promedio')  
    text = Paragraph(text="""El polígono de color rosa es el promedio de liga de acuerdo a la posición. Se considera jugadores con +500 min.
    El polígono azul muestra las estadíticas del jugador.""",
    width=710, height=35)
    textb = Paragraph(text="""
    Min=  Minutos/90 ----- SH90=  Tiros por 90 min ----- KP90 = Pases claves por 90 min ----- xG, xA, xG90, xA90 = Valores esperados""", 
    width = 1000, height = 50)
    text1 = Paragraph(text = """ Se muestra un valor deseado si la diferencia es negativa, entre el valor esperado y las asistencias o goles realizados en la temporada 19/20.
    """,width=1000, height=30)
    text2 = Paragraph(text = """
    """,width=200, height=15)
    text3 = Paragraph(text = """ Se puede pulsar el país para conocer la magnitud de tweets.
    """,width=800, height=35)
    st.bokeh_chart(column(widget, p, text, textb, dot, text1, p1, text2, p2, text3))
    st.header('Valoraciones')
    ana1 = Paragraph(text="""
    De por sí algunos de los datos ya hablan por si solos. Nos apoyan en nuestras creencias acerca de estos jugadores o cuestionan y terminan ayudándonos a reformular nuestras opiniones.
    Por ejemplo, se puede decir que Jack Grealish, Ferran Torres, Adama Traoré o Kai Havertz son jugadores que están rindiendo a gran nivel dentro de sus ligas y al mismo tiempo son demandados por los aficionados.
    Con las métricas, estádisticas y los datos recolectados se puede analizar y poner en uso estos para crear una evaluación de sus valoraciones en el mercado. Para esto utilizamos un sistema de valoración ponderado y poder asignar a cada jugador una categoría dentro de un sistema de recomendación en base a 5 estrellas.
    Los factores a evaluar serán: nivel de juego sobre el promedio de la liga, balance de los valores esperados, el nivel de la liga donde jugo la última temporada el jugador, estimación por parte de los fans según el "sentiment value" de los tweets, el nivel del equipo donde compitío el jugador en la última temporada y el sistema por último considera la edad del jugador.
    Cada factor tiene un peso diferente en sistema de recomendación que se describe en este cuadro.
    """,
    width=650, height=210)
    st.bokeh_chart(ana1)
    ###tablaponde
    contenido = {'Categorías': ['Nivel de juego', 'Balance de valores esperados', 'Nivel de la liga', 'Estimación por fans',  'Nivel del club', 'Edad'], 
                 'Descripción': ['Se considera el número de métricas del jugador que están por encima del promedio.', 'Se toma en cuenta la diferencia en los balances si es deseada y la magnitud. (xG - Goles y xA - Asistencias)', 'De escala 1-10 se evalua el nivel de la liga donde el jugador jugo la última temporada.', 'El sentiment value de los tweets sobre el jugador (promedio por día).', "De escala 1-10 se evalua el nivel del club donde el jugador jugo la última temporada.", "La edad del jugador." ],
                 "Valores": ['45%', '30%', "10%", "5%", "5%", "5%"]}
    factores = pd.DataFrame(contenido)
    st.table(factores)

    ana2 = Div(text="""
    <b>El sistema de ponderación tiene el objetivo de estimar la valoración del jugador y darnos a conocer si un jugador está sobrevalorado, está con una apreciación justa, ó su valor de mercado está subestimado. 
    Los valores de mercado serán estimados según el sistema de recomendación que se detalla a continuación.</b>
    """,
    width=630, height=90)
    
    st.bokeh_chart(ana2)
     ###tablaestrellas
    recomendaciones = {'Recomendación': ['5', '4.5', '4', '3.5', '3', '2.5'],
                       'Valor en millones (€)' : [' > 125 ', '90 - 125', '60 - 89', '40 - 59', '20  - 39', '< 20']}
    estrellas = pd.DataFrame(recomendaciones)
    st.table(estrellas)
   
    st.header('Resultados') 

    ana3 = Paragraph(text="""
    Con las condiciones del sistemad de recomendación ponderado podemos evaluar a los jugadores y su valor de mercado de forma objetiva y dentro de un mismo marco de evaluación. Los resultados del sistema de ponderación se encuentran en la columna "Recomendación" y "Valor aprox en millones".
    """,
    width=630, height=80)
    st.bokeh_chart(ana3)
    ###tablarecomendaciones

    contenido = {'Jugador': ['Jack Grealish', 'Adama Traoré', 'Ismaila Sarr', 'Raul Jimenez', 'Houssem Aouar', 'Victor Osimhen', 'Edison Cavani', 'Kai Havertz', 'David Alaba', 'Ferran Torres', 'Lionel Messi', 'Gianluigi Donnarumma'],
                 "Edad" : ['24', '24', '22', '29', '22', '21', '33', '21', '28', '20', '33', '21'],
                 "Equipo (2020/2021)" : ['Aston Villa', 'Wolverhampton', 'Watford', 'Wolverhampton', 'Olympique Lyonnais', 'Napoli', 'Free', 'Chelsea', 'Bayern Munich', 'Manchester City', 'Barcelona', ' AC Milan'],
                 "Valor aprox en millones (€)": ['40', '35', '24.5', '40', '49.5', '70', '20', '80', '65', '27', '112', '60'],
                 "Recomendación" : ['3.5', '3', '2.5','3.5','3','3','2.5','4.5','3','3','5','4.5'],
                 "Valor estimado (sistema de recomendación)":["40-59 millones", "20-39 millones", " 20 millones > ", "40-59 millones", "20-39 millones", "20-39 millones", " 20 millones > ", "90-124 millones", "20-39 millones","20-39 millones"," 125 millones < ", "90-124 millones"],
                 "Estado" : ['valor justo', 'valor justo', "sobrevalorado", "valor justo", "sobrevalorado", "sobrevalorado", "sobrevalorado", "subestimado", "sobrevalorado", "valor justo", "subestimado", "subestimado"]}
    
    recomendacion = pd.DataFrame(contenido)

    st.table(recomendacion)

    ana4 = Paragraph(text="""
    El análisis nos permite observar que un poco más del 40% de los posibles o completados pases de verano que observamos están sobrevalorados. Puede ser una indicación de que los precios de mercado estaban en un pico.
    Es posible que parte de los jugadores observados fueron causa de rumores de traspaso previo a la pandemia que condicionó a muchos clubes a esperar a negociar en futuros escenarios y obtener un mayor poder de negociación.
    Podemos ver en el caso particular de Victor Osimhen donde su transferencia incluso ha sido la cifra record del Napoli, como una indicación de que el mercado mantuvo su característica de valores pico con una alta competencia para un grupo selectivo de jugadores.
    Siendo la muestra seleccionada para este análisis también selectiva donde el objetivo era evaluar en base a varios factores incluyendo los tweets sobre los jugadores,
    se debe tomar en cuenta el rendimiento del mercado de pases fuera de este contexto para sacar mejores conclusiones donde vemos que el mercado de traspasos de por sí se redujo y varias de las posibles transferencias pasaron a ser solo rumores. 
    En este análisis 3 de cada 4 jugadores no han llegado a ser traspasados y la transferencias de solo 1 de los 3 jugadores subestimados o 1 de los 4 jugadores con precio justo han sido completadas.
    Como observación sobre el 'sentiment value' de los tweets, se conoce que estos por lo general tienen un valor neutral ( 4.393 de los 6.042 tweets observados o 73% de los tweets ) y no tienen una magnitud o impacto significante durante el mercado de traspasos. Puede ser un mejor uso al analizar la magnitud del 'sentiment value' en casos o acciones individuales donde el período de recolección es de corta duración.

    """,
    width=630, height=80)
    st.bokeh_chart(ana4)


if __name__ == '__main__':
    main()

### Add Ferran, add Messi, add Gianluigi
### Transfer values text



 