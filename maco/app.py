from re import L
import streamlit as st
from streamlit_folium import folium_static
import folium
import pandas as pd
import numpy as np
import sys

sys.path.append(r'C:\My Folders\Project\Streamlit\m-aco\model.py')

import model as mdl

st.title('Route Generator Beringin Gigantara')
lat, lon = -8.672200, 115.232080
m = folium.Map(location=[lat, lon], zoom_start=16)
data = pd.DataFrame(np.array([['-8.672200, 115.232080','KC DENPASAR RENON'],['-8.673650, 115.177910','DPS SPBU TEUKU UMAR'],
                              ['-8.804410, 115.217450', 'DPS KK RS SURYA HUSADA NUSA DUA'],['-8.696310, 115.161430','THE KERANJANG'],
                              ['-8.678800, 115.215030','BREW ME'],['-8.638530, 115.174110','DENPASAR SPBU KEROBOKAN'],['-8.670293, 115.222341','RS BALI MED'],
                              ['-8.667999, 115.221855','KEJATI DENPASAR'],['-8.669576, 115.214530','LEVEL 21 MALL'],['-8.700071, 115.248513','RS BALI MANDARA'],
                              ['-8.677290, 115.259022','DENPASAR KCP SANUR'],['-8.681980, 115.256310','KC RENON '],['-8.673770, 115.245440','DENPASAR KK ASABRI'],
                              ['-8.666270, 115.216030','KAMPUS WARMADEWA'],['-8.687170, 115.206250','DNPS UD ARYAWAN'],['-8.704497, 115.239257','MOMMY SHOP'],
                              ['-8.681270, 115.195170','SWALAYAN SE MARLBORO'],['-8.678400, 115.212320','RS SURYA HUSADA SANGLAH'],['-8.682570, 115.166420','DENPASAR SUNSET POINT'],
                              ['-8.710333, 115.223647','SENTRIK'],['-8.675620, 115.217880','GRAND SUDIRMAN 1'],['-8.676180, 115.212480','DENPASAR RSUP SANGLAH'],
                              ['-8.675620, 115.217880','GRAND SUDIRMAN 2'],['-8.634555, 115.217370','CELERITY CAFE'],['-8.697055, 115.226736','KAMPUS UNDIKNAS'],
                              ['-8.667733, 115.227311','DNPS KEUANGAN NEGARA'],['-8.669320, 115.215980','DENPASAR DRIVE THRU DEWI SARTIKA'],['-8.669860, 115.208060','ERLANGGA 2'],
                              ['-8.679330, 115.230340','PUB DPS MINI MARKET AYU NADI'],['-8.677660, 115.227410','DNPS KAMPUS UNDIKNAS LAMA'],['-8.668560, 115.228210','TUKAD PAKERISAN']]),columns=['Coordinate','Lokasi'])

def select_data():
    location = st.multiselect('Pilih Lokasi', data['Lokasi'].unique())
    btn = st.button('Proses')
    
    if btn:
        
        selected_data = data[(data['Lokasi'].isin(location))]
        data_latlng = selected_data['Coordinate']
        data_latlng = data_latlng.str.split(',',expand=True)
        # for i in range(len(selected_data)):
        #     folium.Marker(
        #     location=[data_latlng.iloc[i][0],data_latlng.iloc[i][1]],
        #     popup=data.iloc[i]['Lokasi'],
        # ).add_to(m)
        # folium_static(m)

        direct = mdl.directions(selected_data)
        rute = mdl.rute(selected_data)
        
        travel_time = 0
        distance_route = 0
        
        for drc in direct:
            for i, leg in enumerate(drc[0]["legs"]):
                mark_lat = leg['start_location']['lat']
                mark_lon = leg['start_location']['lng']
                folium.Marker([mark_lat, mark_lon]).add_to(m)
            
                for step in leg['steps']:
                    start_lat = step['start_location']['lat']
                    start_lon = step['start_location']['lng']

                    end_lat = step['end_location']['lat']
                    end_lon = step['end_location']['lng']

                    a = [(start_lat, start_lon), (end_lat, end_lon)]
                    folium.PolyLine(a,
                                weight=5,
                                opacity=0.8).add_to(m)
                    
                    html_instructions = step['html_instructions']
                    #st.write(html_instructions)
                travel_time += leg["duration"]["value"]
                distance_route += leg["distance"]["value"]
        folium_static(m)
        st.write('Estimated Travel Time : ', travel_time/3600, 'Hours')
        st.write('Estimated Distance Route : ',distance_route/1000, 'Km')
        st.dataframe(rute)
    
# Create a page dropdown 
page = st.selectbox("Choose your page", ["Home", "Keseluruhan Data", "Perhitungan Seluruh Jarak", "Hasil Rute Seluruh Data"]) 
if page == "Home":
    select_data()
elif page == "Keseluruhan Data":
    st.write(data)
elif page == "Perhitungan Seluruh Jarak":
    dist = mdl.distancematrix(data)
    st.dataframe(dist)
elif page == "Hasil Rute Seluruh Data":
    direct = mdl.directions(data)
    rute = mdl.rute(data)
    
    for drc in direct:
        for i, leg in enumerate(drc[0]["legs"]):
            mark_lat = leg['start_location']['lat']
            mark_lon = leg['start_location']['lng']
            folium.Marker([mark_lat, mark_lon]).add_to(m)
        
            for step in leg['steps']:
                start_lat = step['start_location']['lat']
                start_lon = step['start_location']['lng']

                end_lat = step['end_location']['lat']
                end_lon = step['end_location']['lng']

                a = [(start_lat, start_lon), (end_lat, end_lon)]
                folium.PolyLine(a,
                            weight=5,
                            opacity=0.8).add_to(m)
                
                html_instructions = step['html_instructions']
                #st.write(html_instructions)
    folium_static(m)
    st.dataframe(rute)