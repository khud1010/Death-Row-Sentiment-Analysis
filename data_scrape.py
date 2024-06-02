#%%
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from collections import Counter
import urllib3
urllib3.disable_warnings()

#%%
# Root URL
root_url = "https://www.tdcj.texas.gov/death_row/dr_executed_offenders.html"

bad_url_list = []
# Function to scrape text from URL
def scrape_text(url):
    try: 
        response = requests.get(url, verify=False)
    except:
        print('Request failed')
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        # Extracting text
        text = soup.get_text()
        return text
    else:
        print(f'this url did not work: {url}')
        bad_url_list.append(url)
        return None

inmate_list_raw_text = scrape_text(root_url)

# Taking unneccesary headers and footers from website
inmate_list_cut_text = inmate_list_raw_text[542:-466]
#%%
inmate_records = inmate_list_cut_text.split('\n\n')
#%%
# Create an empty list to store parsed records
parsed_records = []

# Iterate over each record, split it into lines, and extract data
for record in inmate_records:
    lines = record.split('\n')
    # Filter out empty lines
    lines = [line for line in lines if line.strip()]
    parsed_records.append(lines)
#%%
# Extract column names from the first record
columns = parsed_records[1]

#%%
# Create DataFrame from parsed records excluding the header record
inmate_df = pd.DataFrame(parsed_records[1:], columns=columns)
inmate_df.drop(columns=['Link'],axis=1)
# Display the DataFrame
print(f'NUMBER OF INMATES EXECUTED: {len(inmate_df)}')

# %%
names_for_url = []
for index, row in inmate_df.iterrows():
    last_name = row['Last Name']
    first_name =  row['First Name']
    full_name = f'{last_name}{first_name}'.lower().replace(', jr.', '').replace(', sr.', '').replace(', iii', '')
    full_name = ''.join(e for e in full_name if e.isalnum())
    names_for_url.append(full_name)

inmate_df['url_name'] = names_for_url
# %%
last_words_url_pattern = 'https://www.tdcj.texas.gov/death_row/dr_info/$INMATE$last.html'
no_last_statement_url = 'https://www.tdcj.texas.gov/death_row/dr_info/no_last_statement.html'

last_statement_regex = re.compile(r"Last Statement:(.+?)\n\n\n", re.DOTALL)

last_statements_list = []
no_statement_found_list = []
for index, row in inmate_df.iterrows():
    try:
        last_words_url = last_words_url_pattern.replace('$INMATE$', row['url_name'])
        last_words_raw_text = scrape_text(last_words_url)
        if not last_words_raw_text:
            no_last_statement_raw = scrape_text(no_last_statement_url)
            no_last_statement_match = last_statement_regex.search(no_last_statement_raw)
            if no_last_statement_match:
                last_statement = no_last_statement_match.group(1).strip()
                no_statement_found_list.append(f"{row['First Name']} {row['Last Name']}")
        else:
            # Extract execution info
            last_statement_match = last_statement_regex.search(last_words_raw_text)
            if last_statement_match:
                last_statement = last_statement_match.group(1).strip()
    except Exception as e:
        print(e)
        continue

    last_statements_list.append(last_statement)

#%%
inmate_df['last_statement'] = last_statements_list

inmate_df['last_statement'].value_counts()

strings_to_drop = [
    "This inmate declined to make a last statement.",
    "None",
    "No statement given."
]
# Drop rows containing the specified strings
inmate_df = inmate_df[~inmate_df['last_statement'].isin(strings_to_drop)].reset_index()
print(f'NUMBER OF INMATES WITH LAST STATEMENTS: {len(inmate_df)}')

# %%
count = 0
inmate_words_list = []
for idx, row in inmate_df.iterrows():
    words = row['last_statement'].lower().split()
    inmate_words_list.append(words)
    count += len(words)

inmate_df['words'] = inmate_words_list

print(f'COUNT OF ALL WORDS IN THE CORPUS: {count}')

# %%
all_words = [word for sublist in inmate_df['words'] for word in sublist]
counter_obj = Counter(all_words)

print(f'25 MOST COMMON WORDS (PRE-PROCESSING): {counter_obj.most_common(25)}')

#%%
final_df = inmate_df.loc[:, ['Last Name', 'First Name','Age','Date','Race','last_statement']]

#%%
file_path = './inmate_last_words.csv'
final_df.to_csv(file_path)

# %%
