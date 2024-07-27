#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch data for binary questions based on publish and resolve time
def fetch_binary_questions_by_time(publish_time, resolve_time):
    global question_data,questions
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary"
    }
    raw_api_data = requests.get(base, params=params)
    print("API response JSON:", raw_api_data.json())  # Print the API response JSON for debugging
    response_dict = raw_api_data.json()
    #print(f'response_dict, {response_dict}')
    questions = response_dict.get('results', [])
    question_data = []
    
    for question in questions:
        if question.get('possibilities', {}).get('type') == 'binary':
            question_id = question['id']
            question_title = question['title']
            community_prediction = question.get('community_prediction', {})
            prediction_history = community_prediction.get('history', []) 
            for entry in prediction_history:
                entry['question_id'] = question_id
                entry['question_title'] = question_title
                entry['response_date'] = entry['created']
                question_data.append(entry)
    return question_data

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2024, 4, 1)
end_publish_time = datetime(2024, 4, 25)
resolve_time = datetime(2024, 4, 26)

# Fetch binary questions based on publish and resolve time for each day within the specified range
final_data1 = []
current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    print('Now entering loop')
    final_data1 = final_data1 + fetch_binary_questions_by_time(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame



# In[ ]:


final_data


# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch data for binary questions based on publish and resolve time
def fetch_binary_questions_by_time(publish_time, resolve_time):
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary"
    }
    raw_api_data = requests.get(base, params=params)
    response_dict = raw_api_data.json()
    questions = response_dict.get('results', [])
    #print("Questions before filtering for binary type:", questions)  # Print questions before filtering
    return questions
start_publish_time = datetime(2024, 4, 1)
end_publish_time = datetime(2024, 4, 25)
resolve_time = datetime(2024, 4, 26)

# Fetch binary questions based on publish and resolve time for each day within the specified range
final_data = []
current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    print('Now entering loop')
    
    final_data = final_data + fetch_binary_questions_by_time(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)


# In[ ]:


# Filter for binary questions based on the 'type' attribute under 'possibilities'
questions_with_binary_type = [question for question in final_data if question.get('possibilities', {}).get('type') == 'binary']

# Print the filtered questions
print(questions_with_binary_type)


# 

# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_binary_questions(publish_time, resolve_time):
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary"
    }
    
    try:
        raw_api_data = requests.get(base, params=params)
        raw_api_data.raise_for_status()  # This will raise an exception for HTTP error codes
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []  # Return empty list if there's an error

    response_dict = raw_api_data.json()
    results = response_dict.get('results', [])

    extracted_data = []
    for question in results:
        community_prediction = question.get('community_prediction')
        history = community_prediction.get('history', [])

        if question.get('possibilities', {}).get('type') == 'binary':
            extracted_data.append({
                'id': question.get('id'),
                'title': question.get('title'),
                'predictions_num': history.get('np'),
                'Unique_predictors': history.get('nu'),
                'created_time': question.get('created_time'),
                'publish_time': question.get('publish_time'),
                'resolved_time': question.get('resolve_time'),
                'type': question.get('type'),
                'number_of_forecasters': question.get('number_of_forecasters'),
                'prediction_count': question.get('prediction_count'),
                'x1': history.get('x1'),
                'x2': history.get('x2')
                
            })

    return extracted_data

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2023, 4, 1)
end_publish_time = datetime(2023, 4, 25)  # Update year to match current or past dates
resolve_time = datetime(2023, 4, 26)  # Update year to match current or past dates

binary_questions_data = []

current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    binary_questions_data += fetch_binary_questions(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(binary_questions_data)
print(df)  # Display the dataframe to ensure it's populated


# # Final_fetching!!

# In[3]:


import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_binary_questions(publish_time, resolve_time):
    Username: mp55
    Token: "e1390a56eedbcb8fe1fd998b38091c59d0aa7ead"
    base = "https://www.metaculus.com/api2/questions/"  #Questions url
    
    #Change Publish times and resolve times with correct format
    
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'), 
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary" #Binary Questions
    }
    #headers = {
        #"Authorization": f"Token {Token}"
    #}
    
    try:
        raw_api_data = requests.get(base, params=params) # We request the questions
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []  # Return empty list if there's an error

    response_dict = raw_api_data.json() #We turn it to json for working with it 
    
    #start_time = response_dict.get('unweighted_community_prediction_v2', []).get('start_time',[])
    
    results = response_dict.get('results', [])  # We enter the results logs
    extracted_data = []
    for question in results:
        community_prediction = question.get('community_prediction', {}) #You extract the individual predictions
        history = community_prediction.get('history', []) #You take all the history
        unweighted_community_prediction_v2 = community_prediction.get('unweighted_community_prediction_v2', []) # 

        if question.get('possibilities', {}).get('type') == 'binary':
            for h in history:
                extracted_data.append({
                            'id': question.get('id'),
                            'title': question.get('title'),
                            'predictions_num': h.get('np'),
                            'Unique_predictors': h.get('nu'),
                            'Response_date': h.get('t'),
                            'publish_time': question.get('publish_time'),
                            'resolved_time': question.get('resolve_time'),
                            'number_of_forecasters': question.get('number_of_forecasters'),
                            'prediction_count': question.get('prediction_count'),
                            'q1': h.get('x1').get('q1'),
                            'q2': h.get('x1').get('q2'),
                            'q3': h.get('x1').get('q3'),
                            'avg': h.get('x2').get('avg'),
                            'w_avg': h.get('x2').get('weighted_avg'),
                            'var': h.get('x2').get('var'), 
                            'Result': question.get('resolution'),
                            'weekly_movement': h.get('weekly_movement')
                })

    return extracted_data
                    

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2022, 4, 18)
end_publish_time = datetime(2022, 4, 20)  # Update year to match current or past dates
resolve_time = datetime(2023, 1, 1)  # Update year to match current or past dates

binary_questions_data = []

current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    binary_questions_data += fetch_binary_questions(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(binary_questions_data)
print(df)  # Display the dataframe to ensure it's populated
import datetime
df['Response_date'] = df['Response_date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x))


# In[23]:


"""
import requests
import pandas as pd
from datetime import datetime, timedelta

def fetch_binary_questions(publish_time, resolve_time, token):
   

    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary"
    }

    headers = {
        "Authorization": f"Token {token}"
    }

    try:
        raw_api_data = requests.get(base, params=params, headers=headers)
        raw_api_data.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return []

    response_dict = raw_api_data.json()
    results = response_dict.get('results', [])

    extracted_data = []
    for question in results:
        community_prediction = question.get('community_prediction', {})
        history = community_prediction.get('history', [])

        if question.get('possibilities', {}).get('type') == 'binary':
            for h in history:
                entry = {
                    'id': question.get('id'),
                    'title': question.get('title'),
                    'predictions_num': h.get('np'),
                    'Unique_predictors': h.get('nu'),
                    'Response_date': h.get('t'),
                    'publish_time': question.get('publish_time'),
                    'resolved_time': question.get('resolve_time'),
                    'number_of_forecasters': question.get('number_of_forecasters'),
                    'prediction_count': question.get('prediction_count'),
                    'q1': h.get('x1').get('q1'),
                    'q2': h.get('x1').get('q2'),
                    'q3': h.get('x1').get('q3'),
                    'avg': h.get('x2').get('avg'),
                    'w_avg': h.get('x2').get('weighted_avg'),
                    'var': h.get('x2').get('var'),
                    'Result': question.get('resolution'),
                    'weekly_movement': h.get('weekly_movement')
                }
                extracted_data.append(entry)

        metaculus_prediction = question.get('metaculus_prediction', {})
        #history_m = metaculus_prediction.get('history', [])

        if question.get('possibilities', {}).get('type') == 'binary':
            for h_m in metaculus_prediction:
                entry_m = {
                    'id': question.get('id'),
                    'title': question.get('title'),
                    'predictions_num_m': h_m.get('np'),
                    'Unique_predictors_m': h_m.get('nu'),
                    'q1_m': h_m.get('x1').get('q1'),
                    'q2_m': h_m.get('x1').get('q2'),
                    'q3_m': h_m.get('x1').get('q3'),
                    'avg_m': h_m.get('x2').get('avg'),
                    'w_avg_m': h_m.get('x2').get('weighted_avg'),
                    'var_m': h_m.get('x2').get('var'),
                    'weekly_movement': h_m.get('weekly_movement')
                }
                extracted_data.append(entry_m)

    return extracted_data

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2021, 5, 23)
end_publish_time = datetime(2021, 5, 25)  # Update year to match current or past dates
resolve_time = datetime(2023, 4, 26)  # Update year to match current or past dates

# Replace with your actual Metaculus API token
token = "e1390a56eedbcb8fe1fd998b38091c59d0aa7ead"

binary_questions_data = []

current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    binary_questions_data += fetch_binary_questions(current_publish_time, resolve_time, token)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(binary_questions_data)
print(df)  # Display the dataframe to ensure it's populated

# Ensure the 'Response_date' is correctly formatted
df['Response_date'] = pd.to_datetime(df['Response_date'], unit='s', utc=True)

# Debugging step: Check for weekly movement data
print(df[['id', 'weekly_movement']].head())

"""


# In[14]:





# In[4]:


import datetime
df['Response_date'] = df['Response_date'].apply(lambda x: datetime.datetime.utcfromtimestamp(x.timestamp()))
import pandas as pd 
df_path = 'C:/Users/maorb/CSVs/MetaculusMediaTest.csv'
df.to_csv(df_path, index=False)


# In[ ]:


import pandas as pd 
df_path = 'C:/Users/maorb/CSVs/MetaculusNewest.csv'
df.to_csv(df_path, index=False)


# ## visualization

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load your data
df = pd.read_csv( r"C:\Users\maorb\CSVs\MetaculusNewest.csv", parse_dates=['Response_date'])  # make sure to replace 'your_data.csv' and 'date_column_name'

# Step 2: Ensure the date column is datetime type
df['Response_date'] = pd.to_datetime(df['Response_date'], errors='coerce')

# Step 3: Set the date column as the index (optional, but often helpful for time series data)
df.set_index('Response_date', inplace=True)

# Step 4: Sort the DataFrame by the date index
df.sort_index(inplace=True)

# Step 5: Plotting
plt.figure(figsize=(10, 6))  # You can adjust the size of the figure
sns.lineplot(data=df, x=df.index, y='q2')  # Replace 'value_column_name' with the name of your data column

plt.title('Time Series Plot')
plt.xlabel('Date')
plt.ylabel('Values')

plt.show()


# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta
# Function to fetch data for binary questions based on publish and resolve time
def fetch_binary_questions(publish_time, resolve_time):
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary"
    }
    raw_api_data = requests.get(base, params=params)
    response_dict = raw_api_data.json()
    results = response_dict.get('results', [])
    
    extracted_data = []
    for question in results:
        if question.get('type') == 'binary':
            extracted_data.append({
                'id': question.get('id', None),
                'title': question.get('title', None),
                'created_time': question.get('created_time', None),
                'publish_time': question.get('publish_time', None),
                'resolved_time': question.get('resolve_time', None),
                'type': question.get('type', None),
                'number_of_forecasters': question.get('number_of_forecasters', None),
                'prediction_count': question.get('prediction_count', None),
                'history_x': question.get('community_prediction', {}).get('history', [])[0].get('x', None),
                'history_x1': question.get('community_prediction', {}).get('history', [])[1].get('x', None),
                'history_x2': question.get('community_prediction', {}).get('history', [])[2].get('x', None),
            })
    
    return extracted_data
start_publish_time = datetime(2024, 4, 1)
end_publish_time = datetime(2024, 4, 25)
resolve_time = datetime(2024, 4, 26)
binary_questions_data = []
while current_publish_time <= end_publish_time:
    # Fetch binary questions
    binary_questions_data = append.fetch_binary_questions(current_publish_time,resolve_time)

# Convert list of dictionaries to DataFrame
    df = pd.DataFrame(binary_questions_data)



# In[ ]:


import requests
import pandas as pd
from datetime import datetime, timedelta

# Function to fetch data for binary questions based on publish and resolve time
def fetch_binary_questions(publish_time, resolve_time):
    global extracted_data
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d'),
        "type": "binary"
    }
    raw_api_data = requests.get(base, params=params)
    response_dict = raw_api_data.json()
    results = response_dict.get('results', [])
    
    extracted_data = []
    for question in results:
        if question.get('type') == 'binary':
            extracted_data.append({
                'id': question.get('id', None),
                'title': question.get('title', None),
                'created_time': question.get('created_time', None),
                'publish_time': question.get('publish_time', None),
                'resolved_time': question.get('resolve_time', None),
                'type': question.get('type', None),
                'number_of_forecasters': question.get('number_of_forecasters', None),
                'prediction_count': question.get('prediction_count', None),
                'history_x': question.get('community_prediction', {}).get('history', [])[0].get('x', None),
                'history_x1': question.get('community_prediction', {}).get('history', [])[1].get('x', None),
                'history_x2': question.get('community_prediction', {}).get('history', [])[2].get('x', None),
            })
    
    return extracted_data

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2024, 4, 1)
end_publish_time = datetime(2024, 4, 25)
resolve_time = datetime(2024, 4, 26)

binary_questions_data = []

# Initialize current_publish_time before the loop
current_publish_time = start_publish_time

while current_publish_time <= end_publish_time:
    # Fetch binary questions and concatenate the lists
    binary_questions_data = binary_questions_data + fetch_binary_questions(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(binary_questions_data)



# In[ ]:


import requests
from datetime import datetime, timedelta
import pandas as pd

# Function to fetch data for questions based on publish and resolve time
def fetch_questions_by_time(publish_time, resolve_time):
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d')
    }
    response_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    raw_api_data = requests.get(base, params=params)
    response_dict = raw_api_data.json()
    questions = response_dict['results']
    question_data = []
    for question in questions:
        if question.get('possibilities', {}).get('type') == 'binary':
            question_id = question['id']
            question_title = question['title']
            prediction_history = question['community_prediction']['history']
            for entry in prediction_history:
                entry['question_id'] = question_id
                entry['question_title'] = question_title
                entry['response_date'] = publish_time
                question_data.append(entry)
        return question_data

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2024, 4, 1)
end_publish_time = datetime(2024, 4, 25)  # Adjusted to 04.04.2023 as specified
resolve_time = datetime(2024, 4, 26)  # Adjusted to 01.01.2024 as specified

# Fetch questions based on publish and resolve time for each day within the specified range
final_data = []
current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    final_data = final_data + fetch_questions_by_time(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame
df = pd.DataFrame(final_data)




# In[ ]:


binary_questions_data


# In[ ]:


import requests
from datetime import datetime, timedelta

# Function to fetch data for questions based on publish and resolve time
def fetch_questions_by_time(publish_time, resolve_time):
    base = "https://www.metaculus.com/api2/questions/"
    params = {
        "publish_time__gt": publish_time.strftime('%Y-%m-%d'),  # Corrected date formatting
        "publish_time__lt": (publish_time + timedelta(days=1)).strftime('%Y-%m-%d'),
        "resolve_time__lt": resolve_time.strftime('%Y-%m-%d')
    }
    response_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    raw_api_data = requests.get(base, params=params)
    response_dict = raw_api_data.json()
    questions = response_dict['results']
    question_data = []
    for question in questions:
        if question.get('possibilities', {}).get('type') == 'binary':
            question_id = question['id']
            question_title = question['title']
            prediction_history = question['community_prediction']['history']
            for entry in prediction_history:
                entry['question_id'] = question_id
                entry['question_title'] = question_title
                entry['response_date'] = publish_time__gt
                question_data.append(entry)
        return question_data

# Define start and end dates for publish and resolve time
start_publish_time = datetime(2024, 4, 1)
end_publish_time = datetime(2024, 4, 25)  # Adjusted to 04.04.2023 as specified
resolve_time = datetime(2024, 4, 26)  # Adjusted to 01.01.2024 as specified

# Fetch questions based on publish and resolve time for each day within the specified range
final_data = []
current_publish_time = start_publish_time
while current_publish_time <= end_publish_time:
    final_data = final_data + fetch_questions_by_time(current_publish_time, resolve_time)
    current_publish_time += timedelta(days=1)

# Convert list of dictionaries to DataFrame
# Further code for converting to DataFrame and handling the data goes here


# In[ ]:


final_data


# In[ ]:


import pandas as pd 
df = pd.DataFrame(final_data)
df_path = 'C:/Users/maorb/CSVs/MetaculusNewest.csv'
df.to_csv(df_path, index=False)


# In[ ]:


1711985485.67093 1711985946.940148  #7 equal numbers


# In[ ]:


from openai import OpenAI
import pandas as pd
reference_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19HAHR.JPG"
target_image_path = r"C:\Users\maorb\Desktop\Experiment Builder\KDEF\BM19\BM19AFHR.JPG"
client = OpenAI(
    api_key = "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" # API key goes here
)

gpt_raw_responses = list() # list of raw responses by GPT
assistant = client.beta.assistants.create(
    name="Goal Attainment Predictor",
    instructions="Please adjust brightness and contrast of target image path to match reference image as much as possible",
    model="gpt-4-0125-preview"
)

# # if you've run this code before, you can retrieve the assistant you have created using its ID. Example:
# assistant = client.beta.assistants.retrieve('asst_0DW4sJjSLt9OOAL9O8drQs5T')


gpt_response = messages.data[0].content[0].text.value # take the last message from the thread, which should be GPT's response, unless there was an error
gpt_raw_responses.append(gpt_response) # save response
print("Response {} out of {} received.".format(counter, total_descriptions))









# In[1]:


import os
import base64

from openai import OpenAI
file_path = r"C:\Users\maorb\CSVs\MetaculusNewest.csv"
with open(file_path, 'rb') as file:
    file_content = file.read()

# Encode the file content (if needed)
encoded_content = base64.b64encode(file_content).decode('utf-8')
client = OpenAI(
    # This is the default and can be omitted
    api_key="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"
)


# Generate Python code based on the prompt
response = client.chat.completions.create(
     messages=[
        {
            "role": "user",
            "file": encoded_content,
            "content": """ 
Hey brother, dimension wise the code doesn't work can you fix it ?
def W_i_calc(X_i,S):
    #Let's convert W_i to a matrix of orthogonal matrices: W_{i} = U_{i} @ V_{i}^T where U_{i} @ D_{i} @ V_{i}^T = SVD(X_{i}S^T)
    X_S_T = np.dot(X_i,S.T)
    X_S_T = X_S_T[:,np.newaxis]
    # Perform SVD
    U_i, Sigma, V_i_T = np.linalg.svd(X_S_T, full_matrices=False)
    
    # Calculate W_i as U_i V_i^T
    W_i = np.dot(U_i, V_i_T)
    
    return W_i



def SRM(X_i):
    m = len(X_i)
    W_i = np.ones(m) #Initialize W_i as a matrix of ones
    W_i = W_i[:,np.newaxis] #Convert W_i to a column vector
    #S = 1/m * sum(W_i.T @ X_i)#S = 1/m * sum(W_i * X_i)
    S = (1 / m) * np.dot(W_i.T, X_i)
    S = S[:,np.newaxis].T
    W_i_new = W_i_calc(X_i,S) #Calculate W_i_new
    #We can find the argmin W_i of the function ||X - W_i @ S||^2 by finding the argmin of ||X - U_i @ D_i @ V_i^T @ S||^2 :
    A = np.linalg.norm(X_i - np.dot(W_i_new,S.T), 'fro')**2 #We'll find the argmin W_i of this function
    arg_min = np.argmin(A) 
    return arg_min
    
Note: THis is an SRM model, maybe iif you fix suff try to stay rtrue to the original
"""
        }
    ],
    model="gpt-4-turbo",
)


# Extract the generated Python code from the response
#generated_code = response.choices[0].text.strip()

# Print the generated Python code
#print("Generated Python code:")
#print(generated_code)


# In[ ]:





# In[ ]:


response


# In[ ]:


import numpy as np
A = 0.94+ 0.15 + 26**2 * 0.00134 - 2 *0.0286 -2*26*0.0348 +52 *0.0002
np.sqrt(A)


# In[ ]:


(1/16)**2 * (1-81/256) + (1/16)*(15/16) *(81 / 256) *2 + (15/16)**2 *(1-81/256)+ \
(15/16)**2 *(81/256) +2*(1/16)*15/16*(1-81/256) + (1/16)**2 * (81/256)

(2/3) * (1/16)**2 * (1-81/256) + \
0.5 * (1/16)*(15/16) *(81 / 256) *2 + \
2*(1/16)*15/16*(1-81/256) + \
(1/16)**2 * (81/256)


# In[ ]:


for i in range(0,5,2):
    print(i)


# In[ ]:


import math

# Define inf and nan
inf = float('inf')
nan = float('nan')

# Sum inf and nan
result = inf + nan
result

