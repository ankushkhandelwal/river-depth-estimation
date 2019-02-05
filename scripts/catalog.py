#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Prerequisites: python 3.6 or later
import requests
import json
import sys
#import uuid


# In[2]:


# For real interactions with the data catalog, use api.mint-data-catalog.org
url = "https://sandbox.mint-data-catalog.org"


# In[3]:


# When you register datasets or resources, we require you to pass a "provenance_id". This a unique id associated
# with your account so that we can keep track of who is adding things to the data catalog. For sandboxed interactions
# with the data catalog api, please use this provenance_id:
#provenance_id = "e8287ea4-e6f2-47aa-8bfc-0c22852735c8"
provenance_id = "5656c93c-895e-41b7-aa8d-09053e4ae4d1"


# In[4]:


# Step 1: Get session token to use the API
resp = requests.get(url + '/get_session_token').json()
#print(resp)
api_key = resp['X-Api-Key']

request_headers = {
    'Content-Type': "application/json",
    'X-Api-Key': api_key
}


# In[ ]:


# Our setup
#
# Recall from the data catalog primer (https://docs.google.com/document/d/1I3CjYB-GDdFTZO-dHsHB0B5f0iEbzHnr8QHSKN5k3Sc/edit#heading=h.crwrtnf2ch1h) 
# that a *dataset* is logical grouping of data about specific *variables* contained in one or more *resources*
#
# To make the above statement more concrete, we will interactively go through the process of registering a toy
# dataset in the data catalog in order to make it available for others.
#
# Let's say I have a dataset called "Temperature recorded outside my house" in which every day I note the
# temperature outside my apartment in the morning, afternoon, and evening. I then record those data points in a 
# csv file temp_records_YYYY_MM_DD.csv that looks like:
#
# |        Time         | Temperature |
# -------------------------------------
# | 2018-01-01T07:34:40 |    23       |
# | 2018-01-01T12:15:28 |    32       |
# | 2018-01-01T20:56:15 |    26       |
#
# Note that each file contains data for a single day only. 
#
# In this example, my dataset would be "Temperature recorded outside my house", variables would be 
# "Time" and "Temperature", and each csv file would be a resource associated with the dataset. In addition,
# since each file contains both of the variables in our dataset, each resource will be associated with both variables


# In[5]:


# Now, *I* know that what I refer to as "Temperature" is actually the air temperature recorded in F, but my CSV
# files have no mention of the fact. If you just look at the file without any context, it's unclear what it is that 
# is being recorded. Temperature of what? In what units? C, F, K?
#
# In order to disambiguate those variable names, we require that each variable in your dataset to be associated
# with one or more *standard* variables. What makes a variable name "standard" is that it is a part of some ontology,
# so that anyone can examine that ontology and see for themselves semantic meaning of the variable.
# Most of our current datasets are mapped to standard names defined by the GSN ontology. 
# But you are not forced to map your variables to GSN names. Data catalog allows you to register your own set 
# of standard_variable_names. The only requirement for now is that those standard names are associated with an
# ontology whose schema is publicly available.
# 
# @param[name] standard variable name (aka label)
# @param[ontology] name of the ontology where standard variables are defined
# @param[uri] uri of standard variable name (note that this is full uri, which includes the ontology)
standard_variable_defs = {
    "standard_variables": [
        {
            "name": "Time_Standard_Variable",
            "ontology": "MyOntology",
            "uri": "http://my_ontology_uri.org/standard_names/time_standard_variable"
        },
        {
            "name": "River_Width_Standard_Variable",
            "ontology": "MyOntology",
            "uri": "http://my_ontology_uri.org/standard_names/river_width_standard_variable"
        }
    ]
}

resp = requests.post(url + '/knowledge_graph/register_standard_variables', 
                    headers=request_headers, 
                    json=standard_variable_defs).json()

#print(resp)

# If request is successful, it will return 'result': 'success' along with a list of registered standard variables
# and their record_ids. Those record_ids are unique identifiers (UUID) and you will need them down the road to 
# register variables
if resp['result'] == 'success':
    records = resp['standard_variables']
    time_standard_variable = next(record for record in records if record["name"] == "Time_Standard_Variable")
    river_width_standard_variable = next(record for record in records if record["name"] == "River_Width_Standard_Variable")
    
#    print(f"Time Standard Variable: {time_standard_variable}")
#    print(f"River Width Standard Variable: {river_width_standard_variable}")


# In[6]:


# After you are satisfied that all relevant standard variables are in the data catalog (usually it's a one-time thing),
# you can proceed to register datasets, variables, and resources


'''
# In[20]:


# Step 1: Register datasets
dataset_uuid = "" # This is optional; if not given, it will be auto-generated

dataset_defs = {
    "datasets": [
        {
            "provenance_id": provenance_id,
            "metadata": {"any_additional_metadata": "content"},
            "description": "River Width variation of a stream in South Sudan",
            "name": "River Width Variation"
        }
    ]
}

resp = requests.post(f"{url}/datasets/register_datasets", 
                                        headers=request_headers,
                                        json=dataset_defs).json()


if resp['result'] == 'success':
    datasets = resp["datasets"]
    print(datasets)
    
    dataset_record = next(record for record in datasets if record["name"] == "River Width Variation")
    dataset_record_id = dataset_record["record_id"]
    

# In[22]:


# Step 2: Register variables

# Again, these ids are optional and will be auto-generated if not given. They are included here in order
# to make requests indempotent (so that new records aren't beeing generated every time this code block is run)

#time_variable_record_id = '9358af57-192f-4cc3-9bee-837e76819674'
#river_width_variable_record_id = 'c22deb3b-ebda-48cb-950a-2f4f00498197'

variable_defs = {
    "variables": [
        {
            "dataset_id": dataset_record_id,
            "name": "Time",
            "metadata": {
                "units": "ISO8601_datetime"
            },
            "standard_variable_ids": [
                time_standard_variable["record_id"]
            ]
        },
        {
            "dataset_id": dataset_record_id,
            "name": "River Width",
            "metadata": {
                "units": "meters"
            },
            "standard_variable_ids": [
                river_width_standard_variable["record_id"]
            ]
        }
    ]
}

resp = requests.post(f"{url}/datasets/register_variables", 
                                        headers=request_headers,
                                        json=variable_defs).json()


if resp['result'] == 'success':
    variables = resp["variables"]
    
    time_variable = next(record for record in variables if record["name"] == "Time")
    river_width_variable = next(record for record in variables if record["name"] == "River Width")
    
    print(f"Time Variable: {time_variable}")
    print(f"River Width Variable: {river_width_variable}")


# In[8]:
'''

dataset_record_id = "3094dcb9-1fc8-452a-8ade-cab18545c93f"
time_variable_record_id = "59c46756-a4d4-4583-83a0-87c77ab4fa14"
river_width_variable_record_id = "33ed8091-4d54-413e-bff6-2e1d1fee6fad"


# In[10]:


# Step 3: Register resources
# Assume that I host my datasets files on www.my_domain.com/storage
data_storage_url = "https://raw.githubusercontent.com/ankushkhandelwal/river-depth-estimation/master/CSVs"

# Also, assume that I've collected 2 days worth of data
# in temp_records_2018_01_01.csv and temp_records_2018_01_02.csv. 
file_1_name = sys.argv[1]
#file_2_name = "GEETest3-2.csv"

# and uploaded them to my remote storage location
file_1_data_url = data_storage_url + '/' + file_1_name
#file_2_data_url = f"{data_storage_url}/{file_2_name}"

# Similar to dataset and variable registrations, we are going to generate unique resource record_ids to 
# make these requests repeatable without creating new records. But remember, these will 

#file_1_record_id = "dd52e66b-3149-4d46-8f8e-a18e46136e55"
#file_2_record_id = "25916ccf-d108-4187-b243-2b257ce67fa5"

# Next, let's say that my house is somewhere in LA, 
# defined by the following bounding box (where x refers to longitude and y refers to latitude)
# x_min:  33.9605286
# y_min: -118.4253354
# x_max: 33.9895077
# y_max: -118.4093589

# If I want my resources to be searchable by time range, I can "annotate" each resource with corresponding 
# temporal coverage. That way, when someone searches for any datasets that contain "Temperature_Standard_Variable"
# for January 01 2018, my file_1_name will be returned, along with the data url, and the users will be able to 
# download it easily. Note that temporal coverage must have "start_time" and "end_time" and must follow ISO 8601 
# datetime format YYYY-MM-DDTHH:mm:ss

file_1_temporal_coverage = {
    "start_time": sys.argv[2],
    "end_time": sys.argv[3]
}

#file_2_temporal_coverage = {
#    "start_time": "2016-01-02T00:00:00",
#    "end_time": "2016-12-31T23:59:59"
#}

# Similarly, if I want my datasets to be searchable by location, I can annotate them with spatial coverage. Since
# they are from the same location, we can reuse the same values. Things to note here are required "type" and "value"
# parameters
file_1_spatial_coverage = {
    "type": "BoundingBox",
    "value": {
        "xmin": float(sys.argv[5]),
        "ymin": float(sys.argv[4]),
        "xmax": float(sys.argv[7]),
        "ymax": float(sys.argv[6])
    }
}

#file_2_spatial_coverage = {
#    "type": "BoundingBox",
#    "value": {
#        "xmin": 8.00388159624,
#        "ymin": 26.4215067661,
#        "xmax": 8.00590890856,
#        "ymax": 26.423481421
#    }
#}

# Now we can build our resource definitions

resource_defs = {
    "resources": [
        {
            "dataset_id": dataset_record_id,
            "provenance_id": provenance_id,
            "variable_ids": [
                time_variable_record_id,
                river_width_variable_record_id
            ],
            "name": file_1_name,
            "resource_type": "csv",
            "data_url": file_1_data_url,
            "metadata": {
                "spatial_coverage": file_1_spatial_coverage,
                "temporal_coverage": file_1_temporal_coverage
            },
            "layout": {}
        }
    ]
}

# ... and register them in bulk
resp = requests.post(url + '/datasets/register_resources', 
                                        headers=request_headers,
                                        json=resource_defs).json()


if resp['result'] == 'success':
    resources = resp["resources"]
    
    resource_1 = next(record for record in resources if record["name"] == file_1_name)
#    resource_2 = next(record for record in resources if record["name"] == file_2_name)
    
#    print(f"{file_1_name}: {resource_1}")
#    print(f"{file_2_name}: {resource_2}")


# In[11]:

'''
# Now we can search for data
# 1) Searching by standard_names

search_query_1 = {
    "standard_variable_names__in": ["River_Width_Standard_Variable"]
}

resp = requests.post(f"{url}/datasets/find", 
                                        headers=request_headers,
                                        json=search_query_1).json()
if resp['result'] == 'success':
    found_resources = resp['resources']
    print(f"Found {len(found_resources)} resources")
    print(found_resources)


# In[15]:


# 2) Searching by spatial_coverage

# Bounding box search parameter is a 4-element numeric array (in WGS84 coordinate system) [xmin, ymin, xmax, ymax]
# As a reminder, x is longitude, y is latitude
#27.683673  8.903050
bounding_box = [
    file_1_spatial_coverage["value"]["xmin"]-1, 
    file_1_spatial_coverage["value"]["ymin"]-1,
    file_1_spatial_coverage["value"]["xmax"]+1,
    file_1_spatial_coverage["value"]["ymax"]+1
]

search_query_2 = {
    "spatial_coverage__within": bounding_box
}

resp = requests.post(f"{url}/datasets/find", 
                                        headers=request_headers,
                                        json=search_query_2).json()
if resp['result'] == 'success':
    found_resources = resp['resources']
    print(f"Found {len(found_resources)} resources")
    print(found_resources)


# In[ ]:


# 3) Searching by temporal_coverage and standard_names

# Bounding box search parameter is a 4-element numeric array (in WGS84 coordinate system) [xmin, ymin, xmax, ymax]
# As a reminder, x is longitude, y is latitude
start_time = "2018-01-01T00:00:00"
end_time = "2018-01-01T23:59:59"

search_query_3 = {
    "standard_variable_names__in": [temperature_standard_variable["name"]],
    "start_time__gte": start_time,
    "end_time__lte": end_time
}

resp = requests.post(f"{url}/datasets/find", 
                                        headers=request_headers,
                                        json=search_query_3).json()

if resp['result'] == 'success':
    found_resources = resp['resources']
    print(f"Found {len(found_resources)} resources")
    print(found_resources)
    
    


# In[ ]:

'''


