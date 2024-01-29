#import preinstalled libraries
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import seaborn as sns
from sklearn.decomposition import PCA
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

#extract Kennedy's policies url from online 
kennedy = {
    "economy": "https://www.kennedy24.com/economy", 
    "education": "https://www.kennedy24.com/higher-education",
    "housing": "https://www.kennedy24.com/housing",
    "environment": "https://www.kennedy24.com/environment",
    "immigration": "https://www.kennedy24.com/border",
    "justice": "https://www.kennedy24.com/racial_healing"
}


# function to scrape policies from kennedy
def kennedy_scrape(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an HTTPError for bad responses

        soup = BeautifulSoup(response.content, 'lxml')
        main_element = soup.find("main", {"id": "content"})  # Find the main element

        if main_element:
            text = main_element.get_text()
            return text
        else:
            print(f"Warning: 'main' element with id 'content' not found on {url}")
            return None

    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

# Storing Kennedy's policies
kennedy_policies = {}
for policy, url in kennedy.items():
    #print(f"Scraping {policy} from {url}")
    text = kennedy_scrape(url)

    if text is not None:
        kennedy_policies[policy] = text

# Print Kennedy's policies
#for policy, text in kennedy_policies.items():
#   print(f"Policy: {policy}\n{text}\n{'='*50}\n")
        

#Cornel West
cw_url = "https://www.cornelwest2024.com/platform"

#Jill Stein
js_url = "https://www.jillstein2024.com/principles"

#Claudia de la Cruz
cc_url = "https://votesocialist2024.com/our-program" 


#scrape their pages for their policies
def west_scrape(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        # Find the main element containing policies
        main_element = soup.find("div", {"class": "text-content"})

        if main_element:
            # Find all headings (justice categories)
            headings = main_element.find_all("h3")

            # Initialize a dictionary to store policies
            cornel_west_policies = {}

            # Loop through each heading
            for heading in headings:
                # Extract the original category title
                title = heading.get_text().strip()


                # Find the unordered list under each heading
                unordered_list = heading.find_next("ul")

                # Extract bullet points under each heading
                if unordered_list:
                    list_items = unordered_list.find_all("li")
                    policy_text = "\n".join(li.get_text().strip() for li in list_items)

                    # Store the policies in the dictionary
                    cornel_west_policies[title] = policy_text

            return cornel_west_policies

        else:
            print(f"Warning: 'div' element with class 'text-content' not found on {url}")
            return None

    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return None

# Scrape Cornel West's policies
west_policies = west_scrape(cw_url)

def stein_scrape(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        # Initialize a dictionary to store policies
        jill_stein_policies = {}

        # Find the main element containing policies
        main_element = soup.find('div', class_='accordion')

        if main_element:
            # Find all headings
            headings = main_element.find_all("h2")

            # Loop through each heading
            for heading in headings:
                title = heading.text
                #print(title)

                content = heading.find_next("div", class_="accordion--body").text

                # Store in the dictionary
                jill_stein_policies[title] = content

            return jill_stein_policies

        else:
            print(f"Warning: 'div' element with class 'accordion' not found on {url}")
            return {}

    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return {}

# Scrape Jill Stein's policies
stein_policies = stein_scrape(js_url)


# scrape claudia de la cruz website
def scrape_claudia(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'lxml')

        # Initialize a dictionary to store policies
        claudia_policies = {}

        # Find the main element containing policies
        main_element = soup.find_all('button', class_ = 'accordion-item__click-target')

        if main_element:

            # Loop through each heading
            for button in main_element:
                # Get title
                title = button.find('span', class_='accordion-item__title').text
                
                # Get text content div
                content_div = button.find_next('div', class_='accordion-item__description')
                
                # Extract text 
                content = content_div.text
                
                # Store in the dictionary
                claudia_policies[title] = content

            return claudia_policies

        else:
            print(f"Warning: 'div' element with class 'accordion' not found on {url}")
            return {}

    except requests.RequestException as e:
        print(f"Error scraping {url}: {e}")
        return {}
    
# Scrape claudia policies
cruz_policies = scrape_claudia(cc_url)



# Combine policies into a single dataset
all_policies = {
    'Kennedy': kennedy_policies,
    'Cornel West': west_policies,
    'Jill Stein': stein_policies,
    'Claudia Cruz': cruz_policies
}

# Data Cleaning and Vectorization
documents = [' '.join(policy.values()) for policy in all_policies.values()]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)

# Cosine Similarity
cosine_similarities = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Visualization
sns.heatmap(cosine_similarities, annot=True, xticklabels=all_policies.keys(), yticklabels=all_policies.keys(), cmap="YlGnBu")
plt.title('Cosine Similarity Between Candidates')
plt.show()

# Sentiment Analysis with Subjectivity
sentiments = {candidate: TextBlob(' '.join(policy.values())).sentiment for candidate, policy in all_policies.items()}

# Extract polarity and subjectivity separately
polarities = {candidate: sentiment.polarity for candidate, sentiment in sentiments.items()}
subjectivities = {candidate: sentiment.subjectivity for candidate, sentiment in sentiments.items()}

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Custom colors for bars
bar_colors = ['skyblue', 'pink', 'lightgreen','yellow']

# Bar plot for polarity with custom colors
sns.barplot(x=list(polarities.keys()), y=list(polarities.values()), ax=ax1, palette=bar_colors)
ax1.set_title('Polarity Analysis of Candidates')
ax1.set_ylabel('Polarity')

# Bar plot for subjectivity with custom colors
sns.barplot(x=list(subjectivities.keys()), y=list(subjectivities.values()), ax=ax2, palette=bar_colors)
ax2.set_title('Subjectivity Analysis of Candidates')
ax2.set_ylabel('Subjectivity')

plt.show()

#Generating WordCloud Comparisons
#clean text
stemmer = PorterStemmer()

def clean_text(text):
    tokens = nltk.word_tokenize(text)
    
    # Stem words
    stemmed = [stemmer.stem(w) for w in tokens]  

    # Remove stopwords
    stopwords_set = set(stopwords.words('english'))
    clean_tokens = [w for w in tokens if w not in stopwords_set]

    clean_text =  " ".join(clean_tokens)

    return clean_text

# Concatenate all text 

# List of candidates 
candidates = ["kennedy", "west", "stein", "cruz"]

# Dictionaries of policies
policies = {
    "kennedy": kennedy_policies,
    "west": west_policies,
    "stein": stein_policies,
    "cruz": cruz_policies
}

# Loop through each candidate
for candidate in candidates:

    # Get policies
    policy_dict = policies[candidate]  

    # Concatenate text  
    full_text = ""
    for policy, text in policy_dict.items():
        full_text += text

    # Clean text
    full_text = clean_text(full_text)

    # Save cleaned text
    policies[candidate]["full_text"] = full_text



# Function to generate word cloud and save image
def generate_wordcloud(policy_text, save_path):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(policy_text)
    
    # Display the generated image:
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_path)
    #plt.show()

for candidate in candidates:

    text = policies[candidate]["full_text"]
    save_path = f"{candidate}_wordcloud.png"
    
    generate_wordcloud(text, save_path)

fig, axs = plt.subplots(nrows=1, ncols=len(candidates), figsize=(20, 5))

for i, candidate in enumerate(candidates):

    img = plt.imread(f"{candidate}_wordcloud.png")
    
    axs[i].imshow(img)
    axs[i].set_title(candidate)

plt.tight_layout()
#plt.show()
plt.savefig(f"candidate_wordcloud.png")






