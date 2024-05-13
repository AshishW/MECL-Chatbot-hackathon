import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import spacy
import plotly.graph_objects as go
from pykrige.ok import OrdinaryKriging
from scipy.interpolate import griddata
import base64
import os
import io
import scipy
import plotly
import pykrige



# Get the directory of the current Python file
script_dir = os.path.dirname(__file__)

# Construct the relative path to the CSV file (assuming it's in the same directory)
csv_file_path = os.path.join(script_dir, 'NGDR_Nagpur.csv')
Nagpur_gdf = pd.read_csv(csv_file_path)
# Nagpur_gdf.fillna(0.0, inplace=True)

import textdistance
word_list = ["kriging","concentration","toposheet","interpolation","inverse distance weighted","idw","maximum","minimum","longitude","latitude","aluminum"]
def correct_typos(text, threshold=0.5):
    corrected_text = []
    words = text.split()
    for word in words:
        # Check if the word is misspelled
        suggestions = [w for w in word_list if textdistance.jaccard.normalized_similarity(w, word) >= threshold]
        if suggestions:
            corrected_word = max(suggestions, key=lambda x: textdistance.jaccard.normalized_similarity(x, word))
        else:
            corrected_word = word
        corrected_text.append(corrected_word)
    return ' '.join(corrected_text)



import re

# Dictionary of chemical names and their formulas
chemicals = {
   'silicon dioxide': 'sio2',
    'aluminum oxide': 'al2o3',
    'iron(III) oxide': 'fe2o3',
    'titanium dioxide': 'tio2',
    'calcium oxide': 'cao',
    'magnesium oxide': 'mgo',
    'manganese(II) oxide': 'mno',
    'sodium oxide': 'na2o',
    'potassium oxide': 'k2o',
    'phosphorus pentoxide': 'p2o5',
    'loss on ignition': 'loi',
    'barium': 'ba',
    'gallium': 'ga',
    'scandium': 'sc',
    'vanadium': 'v',
    'thorium': 'th',
    'lead': 'pb',
    'nickel': 'ni',
    'cobalt': 'co',
    'rubidium': 'rb',
    'strontium': 'sr',
    'yttrium': 'y',
    'zirconium': 'zr',
    'niobium': 'nb',
    'chromium': 'cr',
    'copper': 'cu',
    'zinc': 'zn',
    'gold': 'au',
    'lithium': 'li',
    'cesium': 'cs',
    'arsenic': 'as_',
    'antimony': 'sb',
    'bismuth': 'bi',
    'selenium': 'se',
    'silver': 'ag',
    'beryllium': 'be',
    'germanium': 'ge',
    'molybdenum': 'mo',
    'tin': 'sn',
    'lanthanum': 'la',
    'cerium': 'ce',
    'praseodymium': 'pr',
    'neodymium': 'nd',
    'samarium': 'sm',
    'europium': 'eu',
    'terbium': 'tb',
    'gadolinium': 'gd',
    'dysprosium': 'dy',
    'holmium': 'ho',
    'erbium': 'er',
    'thulium': 'tm',
    'ytterbium': 'yb',
    'lutetium': 'lu',
    'hafnium': 'hf',
    'tantalum': 'ta',
    'tungsten': 'w',
    'uranium': 'u',
    'platinum': 'pt',
    'palladium': 'pd',
    'indium': 'in_',
    'fluorine': 'f',
    'tellurium': 'te',
    'thallium': 'tl',
    'mercury': 'hg',
    'cadmium': 'cd'
    # Add more chemicals as needed
}

# Function to extract chemical names and formulas from a sentence
def extract_elements(query):
    # Initialize an empty list to store the extracted elements
    elements = []

    # Convert the sentence to lowercase
    lower_sentence = query.lower()

    # Extract words from the sentence
    words = re.findall(r'\b\w+\b', lower_sentence)

    # Check if any chemical name or formula is present in the sentence
    for chemical, formula in chemicals.items():
        if chemical.lower() in lower_sentence or formula.lower() in words:
            elements.append(formula)

    # Return the list of extracted elements
    return elements


def extract_topo_no(str1):
    toposheet_numbers = []
    toposheet_pattern = r'\b\d+[a-zA-Z]+\d+\b'
    toposheet_numbers = re.findall(toposheet_pattern,str1)
    return toposheet_numbers



def create_kriging_map_from_query(query,df,toposheet_numbers,elements):    
#     combined_krig_results = []
    for toposheet_number in toposheet_numbers:
        for element in elements:
            return generate_kriging_map(df, element, toposheet_number)




def check_variance(z_interp, threshold=0.5):
    """
    Check if the variance of the z_interp array is below a given threshold.
    
    Parameters:
    z_interp (numpy.ndarray): The 2D array of interpolated values.
    threshold (float): The variance threshold to determine if the map will appear blank.
    
    Returns:
    bool: True if the variance is below the threshold, indicating a blank map.
    """
    # Flatten the array to compute the overall variance
    z_flat = z_interp.flatten()
    variance = np.var(z_flat)
    print("VARIANCE: ", variance)
    print(len(z_flat))
    # check if more than 800 values in z_flat are same
    if len(set(z_flat)) < 1000:
        return True
    # Check if the variance is below the threshold
    # if variance < threshold:
    #     return True
    # else:
    #     return False



def generate_kriging_map(df, element, toposheet_number, variogram_model='spherical'):
    # try printing the element, toposheet_number and variogram_model

    try:
         # Function body goes here
     # Filter the DataFrame by the specified toposheet number if provided
        if toposheet_number is not None:
            df = df[df['toposheet'] == toposheet_number]
            
    # Define grid resolution
        gridx = np.linspace(df['longitude'].min(), df['longitude'].max(), 100)
        gridy = np.linspace(df['latitude'].min(), df['latitude'].max(), 100)
        OK = OrdinaryKriging(df['longitude'], df['latitude'], df[element], variogram_model=variogram_model)
        z_interp, ss = OK.execute('grid', gridx, gridy)
        is_blank_map = check_variance(z_interp)
        if is_blank_map:
            return f'Sorry for the inconvenience, the Kriging map of "{element}" for the requested toposheet number cannot be generated. Please stay tuned for updates'
        
        # Create the contour plot
        contour = go.Contour(
            z=z_interp.data,  # 2D array of the heatmap values
            x=gridx,  # X coordinates corresponding to 'z_interp'
            y=gridy,  # Y coordinates corresponding to 'z_interp'
            colorscale='YlOrRd',  # Match the colormap
            showscale=True  # Show the color scale bar
        )
    
    # Create the scatter plot with hover annotations
        scatter = go.Scatter(
            x=df['longitude'],
            y=df['latitude'],
            mode='markers',
            marker=dict(
                color=df[element],
                colorscale='YlOrRd',  # Match the colormap
                showscale=False,  # We already have a color scale from the contour
                line=dict(color='black', width=1)  # Black border around the scatter points
            ),
            text=df[element],  # Text for hover annotations
            hoverinfo='text'  # Show only the text on hover
        )
        max_value = df[element].max()
        max_location = df[df[element] == max_value][['latitude', 'longitude']].iloc[0]
        max_lat, max_lon = max_location['latitude'], max_location['longitude']
            # Minimum value aur uske corresponding latitude, longitude find kar rahe hain
        min_value = df[element].min()
        min_location = df[df[element] == min_value][['latitude', 'longitude']].iloc[0]
        min_lat, min_lon = min_location['latitude'], min_location['longitude']
    
    # Create a figure and add the contour plot
        fig = go.Figure(data=[contour])
    
    # Add the scatter plot on top of the contour plot
#         fig.add_trace(scatter)
    
        # Update layout with title and labels
    #     fig.update_layout(
    #         title=f'Geochemical Kriging Map for {element}' + (f' (Toposheet {toposheet_number})' if toposheet_number else ''),
    #         xaxis_title='Longitude',
    #         yaxis_title='Latitude',
    #         coloraxis_colorbar=dict(title=f'{element} Concentration')
    #     )
        fig.add_annotation(
        text=f"<i>Maximum value (in ppm): <b>{max_value}</b> at longitude <b>{max_lon}</b> and latitude <b>{max_lat}</b></i>",
        xref="paper", yref="paper",
        x=0.5, y=1.2, showarrow=False,
        font=dict(size=14),
        align="center"
        )

        fig.add_annotation(
        text=f"<i>Minimum value (in ppm): <b>{min_value}</b> at longitude <b>{min_lon}</b> and latitude <b>{min_lat}</b></i>",
        xref="paper", yref="paper",
        x=0.5, y=1.1, showarrow=False,
        font=dict(size=14),
        align="center"
        )

    # Add a title to the map
        fig.update_layout(
        title=f"<b>Stream Sediment samples showing {element} Values(ppm)</b>",
        title_x=0.5,  # Center the title
        title_y=0.95,  # Add some space from the top
        margin=dict(t=120)  # Adjust margin to accommodate the annotations and title
        )
      
        data = fig.to_dict()
        layout = fig.to_dict()
        return (data,layout, 'kriging_map')
    except Exception as e:
        return 'Data for requested toposheet number is not available or updated. Please stay tuned for updates!'



# OLD create_idw_map_from_query(query, df)
# def create_idw_map_from_query(query,df):
    t2 = extract_topo_no(query)
    e2 = extract_chemicals(query)
    threshold_percentile = 100   
    for toposheet_number in t2:
        for element in e2:
            max_value = df[element].max()
            max_location = df[df[element] == max_value][['latitude', 'longitude']].iloc[0]
            max_lat, max_lon = max_location['latitude'], max_location['longitude']
            # Minimum value aur uske corresponding latitude, longitude find kar rahe hain
            min_value = df[element].min()
            min_location = df[df[element] == min_value][['latitude', 'longitude']].iloc[0]
            min_lat, min_lon = min_location['latitude'], min_location['longitude']
            return generate_idw_map(df, element, toposheet_number, threshold_percentile) 
            # return generate_idw_map(df, element,max_value, max_location, max_lat, max_lon, min_value, min_location, min_lat, min_lon, toposheet_number, threshold_percentile) 

def create_idw_map_from_query(query,df,toposheet_numbers,elements, threshold_percentile):
#     combined_idw_results = []
    for toposheet_number in toposheet_numbers:
        for element in elements:
            return generate_idw_map(df, element, toposheet_number, threshold_percentile)

        
def generate_idw_map(df, element, toposheet_number, threshold_percentile):
    # Filter the DataFrame by the specified toposheet number
    gdf = df[df['toposheet'] == toposheet_number]
    if (gdf[element] == 0.00).all():
        return(f'Sorry for the inconvenience, the map of "{element}" for requested toposheet number cannot be generated. Please stay tuned for updates!')
    else:
        # Determine Baseline
        baseline = np.median(gdf[element])

        # Calculate Deviation
        deviation = gdf[element] - baseline

        # Statistical Analysis
        std_dev = np.std(deviation)
        percentile_value = np.percentile(deviation, threshold_percentile)

        # Define Anomaly Threshold
        anomaly_threshold = percentile_value

        # Identify Anomalies
        anomalies = gdf[deviation > anomaly_threshold]
        max_value = gdf[element].max()
        max_location = gdf[gdf[element] == max_value][['latitude', 'longitude']].iloc[0]
        max_lat, max_lon = max_location['latitude'], max_location['longitude']
        # Minimum value aur uske corresponding latitude, longitude find kar rahe hain
        min_value = gdf[element].min()
        min_location = gdf[gdf[element] == min_value][['latitude', 'longitude']].iloc[0]
        min_lat, min_lon = min_location['latitude'], min_location['longitude']

        # Create grid coordinates for interpolation
        grid_x, grid_y = np.mgrid[min(gdf['longitude']):max(gdf['longitude']):100j, min(gdf['latitude']):max(gdf['latitude']):100j]

        # Interpolate using IDW
        grid_z = griddata((gdf['longitude'], gdf['latitude']), deviation, (grid_x, grid_y), method='cubic')

        # Create the contour plot
        contour = go.Contour(
            z=grid_z.T,
            x=np.linspace(min(gdf['longitude']), max(gdf['longitude']), 100),
            y=np.linspace(min(gdf['latitude']), max(gdf['latitude']), 100),
            colorscale='Viridis',
            colorbar=dict(title='Deviation from Baseline'),
            showscale=False # Hide the color scale as we have a scatter plot
        )
       
        scatter = go.Scatter(
            x=anomalies['longitude'],
            y=anomalies['latitude'],
            mode='markers',
            marker=dict(
                color='red',
                size=5,
                symbol='circle',
                line=dict(width=1),
                opacity=0.8,
                colorscale='Viridis',
                colorbar=dict(title='ppm'),
                cmin=0,
                cmax=max_value,
                showscale=True
            ),
            name='Anomalies',
            text=anomalies[element]
        )
        # scatter = go.Scatter(
        #     x=anomalies['longitude'],
        #     y=anomalies['latitude'],
        #     mode='markers',
        #     marker=dict(
        #         color='red',
        #         size=5,
        #         symbol='circle',
        #         line=dict(width=1),
        #         opacity=0.8,
        #     ),
        #     name='Anomalies',
        #     text=anomalies[element]
        # )
        # Add annotations for maximum and minimum values
        annotations = [
            dict(
                text=f"<i>Maximum value (in ppm): <b>{max_value}</b> at longitude <b>{max_lon}</b> and latitude <b>{max_lat}</b></i>",
                xref="paper", yref="paper",
                x=0.5, y=1.2, showarrow=False,
                font=dict(size=14),
                align="center"
            ),
            dict(
                text=f"<i>Minimum value (in ppm): <b>{min_value}</b> at longitude <b>{min_lon}</b> and latitude <b>{min_lat}</b></i>",
                xref="paper", yref="paper",
                x=0.5, y=1.1, showarrow=False,
                font=dict(size=14),
                align="center"
            )
        ]
        
        # Define the layout with annotations
        layout = go.Layout(
            width=720,
            height=480,
            annotations=annotations,
            title=f"<b>Stream Sediment samples showing {element} Values in Toposheet {toposheet_number}</b>",
            title_x=0.5,
            title_y=0.95,
            margin=dict(t=120) 
        )

        # Create the figure and plot it
        fig = go.Figure(data=[contour, scatter], layout=layout)
#         fig.show()
        data = fig.to_dict()
        layout = fig.to_dict()
        return (data, layout, 'idw_map')

def find_max_values(query, df,toposheet_numbers,elements):
    for toposheet_number in toposheet_numbers:
        for element in elements:
            max_value = df[element].max()
            max_location = df[df[element] == max_value][['latitude', 'longitude']].iloc[0]
            max_lat, max_lon = max_location['latitude'], max_location['longitude']
            max_value_result = f"For the toposheet {toposheet_number}, the element {element} has maximum PPM value {max_value} at latitude {max_lat} and longitude {max_lon}."
            return max_value_result



def find_min_values(query, df,toposheet_numbers,elements):
#     min_ppm_results = []
    for toposheet_number in toposheet_numbers:
        for element in elements:
            min_value = df[element].min()
            min_location = df[df[element] == min_value][['latitude', 'longitude']].iloc[0]
            min_lat, min_lon = min_location['latitude'], min_location['longitude']    
            # Results ko sentence mein display kar rahe hain
            min_value_result = f"For the toposheet {toposheet_number}, the element {element} has minimum PPM value {min_value} at latitude {min_lat} and longitude {min_lon}."
#             min_ppm_results.append(min_value_result)
            return min_value_result



def find_both_min_max(query, df,toposheet_numbers,elements):

    for toposheet_number in toposheet_numbers:
        for element in elements:
            max_value = df[element].max()
            max_location = df[df[element] == max_value][['latitude', 'longitude']].iloc[0]
            max_lat, max_lon = max_location['latitude'], max_location['longitude']
            min_value = df[element].min()
            min_location = df[df[element] == min_value][['latitude', 'longitude']].iloc[0]
            min_lat, min_lon = min_location['latitude'], min_location['longitude']

            min_max_value_result = f"For the toposheet {toposheet_number}, the element {element} has maximum PPM value {max_value} at latitude {max_lat} and longitude {max_lon}, and has minimum PPM value {min_value} at latitude {min_lat} and longitude {min_lon}."
            return min_max_value_result

# split sub queries ::EXPERIMENTAL
# def split_query_smartly(query):
    element_names = ['silicon dioxide', 'aluminum oxide', 'iron(III) oxide', 'titanium dioxide', 'calcium oxide', 'magnesium oxide', 'manganese(II) oxide', 'sodium oxide', 'potassium oxide', 'phosphorus pentoxide', 'loss on ignition', 'barium', 'gallium', 'scandium', 'vanadium', 'thorium', 'lead', 'nickel', 'cobalt', 'rubidium', 'strontium', 'yttrium', 'zirconium', 'niobium', 'chromium', 'copper', 'zinc', 'gold', 'lithium', 'cesium', 'arsenic', 'antimony', 'bismuth', 'selenium', 'silver', 'beryllium', 'germanium', 'molybdenum', 'tin', 'lanthanum', 'cerium', 'praseodymium', 'neodymium', 'samarium', 'europium', 'terbium', 'gadolinium', 'dysprosium', 'holmium', 'erbium', 'thulium', 'ytterbium', 'lutetium', 'hafnium', 'tantalum', 'tungsten', 'uranium', 'platinum', 'palladium', 'indium', 'fluorine', 'tellurium', 'thallium', 'mercury', 'cadmium']
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")
    
    # Define split words and phrases
    split_words = ["maximum and minimum", "longitude and latitude"]
    additional_split_words = ["also", "display", "create", "produce", "tell", "describe","show"]
    
    # Process the query using spaCy
    doc = nlp(query)
    
    # Initialize variables
    subqueries = []
    subquery = ""
    split_flag = False  # Flag to determine if splitting is allowed
    
    # Iterate over tokens in the query
    for token in doc:
        # Check if token is a split word or phrase
        if any(split_word in token.text.lower() for split_word in split_words):
            split_flag = False
        elif token.text.strip() == "." or token.text.strip() == ". ":
            # Split when encountering a full stop or full stop followed by a space
            if subquery.strip():
                subqueries.append(subquery.strip())
            subquery = ""
            split_flag = False
        elif token.text.strip() in additional_split_words:
            # Split when encountering additional split words
            if subquery.strip():
                subqueries.append(subquery.strip())
            subquery = ""
            split_flag = False
        elif token.text.strip() == "and" and split_flag:
            # Do not split when "and" is encountered between elements
            subquery += token.text_with_ws
        elif token.text.strip() == "and" and subquery.lower().strip() in element_names:
            # Do not split when "and" is encountered after an element name
            subquery += token.text_with_ws
        elif token.text.strip().lower() not in additional_split_words:
            # Append token to subquery if it's not in the additional split words list
            subquery += token.text_with_ws
            split_flag = True  # Set flag to allow splitting after this token
    
    # Append the final subquery if it's not empty and contains more than one word
    if subquery.strip() and len(subquery.split()) > 1:
        subqueries.append(subquery.strip())
    
    return subqueries


def process_subqueries(query, threshold_percentile):
    query = re.sub(r'\b(?:can|could|will|would|shall|should|may|might|must|to)\s+be\b', 'exists', query, flags=re.IGNORECASE)

    word_list = ["kriging", "idw", "max", "min","maximum","minimum"]
    max_found = False
    min_found = False
    both_max_min_found = False
    df = Nagpur_gdf
    toposheet_numbers = extract_topo_no(query)
    elements = extract_elements(query)

    print(toposheet_numbers,'\n')
    print(elements)
    if(len(toposheet_numbers) == 0 or len(elements) == 0) and (not len(toposheet_numbers) == len(elements)):
        return 'Please provide both the toposheet number and element name for further processing of the query.'
    if (len(toposheet_numbers) > 1) or (len(elements) > 1):
        return 'Apologies for any inconvenience. Please note that I can only display data for one element and one toposheet at a time.'
    
    if 'kriging' in query.lower() and 'idw' in query.lower():
        return f'''I apologize for any inconvenience; I can generate only one map at a time.
               You might ask, for example, 'Create a kriging/IDW map for {elements[0]} for toposheet number {toposheet_numbers[0]}'.'''
    else:
        for word in word_list:
            pattern = r'\b{}\b'.format(re.escape(word))
            if re.search(pattern, query.lower()):
                if word == 'kriging':
                    return create_kriging_map_from_query(query, df,toposheet_numbers,elements)
                elif word == 'idw':
                    return create_idw_map_from_query(query, df,toposheet_numbers,elements, threshold_percentile)
                elif word in ['max','maximum']:
                    max_found = True
                elif word in ['min','minimum']:
                    min_found = True
        if max_found and min_found:
            return find_both_min_max(query, df,toposheet_numbers,elements)
        elif max_found:
            return find_max_values(query, df,toposheet_numbers,elements)
        elif min_found:
            return find_min_values(query, df,toposheet_numbers,elements)


def generate_geochemistry_response(query, threshold_percentile=100):
    corrected_sentence = correct_typos(query)
    response = process_subqueries(corrected_sentence, threshold_percentile)
    
    if response == None:
        response = "Sorry, I am unable to respond to this query. I am currently equipped to provide information on Nagpur Geochemistry Toposheet data and can handle one query at a time. Thank you for your understanding."
    data_type = "text"
    if type(response) == tuple and response[2]=='idw_map':
        data_type = "idw_map"
    elif type(response) == tuple and response[2]=='kriging_map':
        data_type = "kriging_map"
    return response, data_type

if __name__ == "__main__":
    generate_geochemistry_response(query="Create a kriging map for copper for the toposheet number 55K14", threshold_percentile=100)
