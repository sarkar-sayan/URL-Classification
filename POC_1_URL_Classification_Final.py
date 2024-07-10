{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMY6xSpVT6e7gLIFMZ4xivP"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RdedyzV66MhL",
        "outputId": "34d86260-5d6f-4ce2-8afd-307bd60d6851"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/drive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/drive')"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "## Code as a User based menu-driven"
      ],
      "metadata": {
        "id": "RR-6jaOYbLVq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "pip install langdetect googletrans==4.0.0-rc1"
      ],
      "metadata": {
        "id": "bS9ilsNObSlE"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import requests\n",
        "from bs4 import BeautifulSoup\n",
        "from random import choice\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "import string\n",
        "from langdetect import detect\n",
        "from googletrans import Translator\n",
        "# Initialize Google Translator\n",
        "translator = Translator()\n",
        "# Initialize Tokenizer\n",
        "vectorizer = TfidfVectorizer()\n",
        "# Download the stopwords\n",
        "nltk.download('stopwords')\n",
        "stop_words = set(stopwords.words('english'))\n",
        "\n",
        "  # List of user-agents for rotation\n",
        "user_agents = [\n",
        "     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',\n",
        "     'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',\n",
        "     'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0',\n",
        "     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',\n",
        "     'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0',\n",
        "     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.2420.81',\n",
        "     'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 OPR/109.0.0.0',]\n",
        "\n",
        "global productive_keywords\n",
        "global non_productive_keywords\n",
        "global prod_domains\n",
        "global nprod_domains\n",
        "\n",
        "productive_keywords = ['study', 'research', 'education', 'work', 'project', 'python']\n",
        "non_productive_keywords = ['game', 'social', 'fun', 'entertainment', 'video', 'reels']\n",
        "\n",
        "prod_domains = ['www.w3schools.com','www.geeksforgeeks.org']\n",
        "nprod_domains = ['www.facebook.com', 'www.instagram.com']"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "vQAIFy0FPvOA",
        "outputId": "49ef897b-583a-426a-9851-208155a0a199"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Enter or delete pre-saved Keywords\n",
        "\n",
        "def keyword_management(productive_keywords, non_productive_keywords):\n",
        "  choice = int(input(\"Enter choice to manage keywords (1 or 2) : 1. Productive  2. Non-productive  3. Display All -> \"))\n",
        "  if(choice == 1):\n",
        "    get_productive(productive_keywords)\n",
        "  elif(choice == 2):\n",
        "    get_nonproductive(non_productive_keywords)\n",
        "  elif choice not in [1, 2]:\n",
        "    print(\"Displaying All....\")\n",
        "    print(\"All Productive Keywords:\", \", \".join(productive_keywords))\n",
        "    print(\"All Non-productive Keywords:\", \", \".join(non_productive_keywords))\n",
        "  return productive_keywords, non_productive_keywords\n",
        "\n",
        "\n",
        "def get_productive(productive_keywords):\n",
        "  ch = int(input(\"Enter choice to manage productive keywords (1 or 2) : 1. Add new  2. Delete old  3. Display all->\"))\n",
        "  if(ch == 1):\n",
        "      c = int(input(\"Add NEW -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter new Productive Keyword: \")\n",
        "        productive_keywords.append(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          key = input(\"\\nEnter keyword to add (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          productive_keywords.append(key)\n",
        "  elif(ch == 2):\n",
        "      c = int(input(\"Delete OLD -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter Productive Keyword to delete: \")\n",
        "        productive_keywords.remove(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          print(f\"\\nCurrent List: {productive_keywords}\")\n",
        "          key = input(\"Enter keyword to remove (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          try:\n",
        "            productive_keywords.remove(key)\n",
        "          except ValueError:\n",
        "            print(f\"'{key}' not found in the list. Try again.\")\n",
        "        print(\"\\nFinal List: \", productive_keywords)\n",
        "  elif(ch == 3):\n",
        "      print(\"All Productive Keywords: \", \", \".join(productive_keywords))\n",
        "\n",
        "\n",
        "def get_nonproductive(non_productive_keywords):\n",
        "  ch = int(input(\"Enter choice to manage non-productive keywords (1 or 2) : 1. Add new  2. Delete old  3. Display all->\"))\n",
        "  if(ch == 1):\n",
        "      c = int(input(\"Add NEW -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter new Non-productive Keyword: \")\n",
        "        non_productive_keywords.append(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          key = input(\"\\nEnter keyword to add (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          non_productive_keywords.append(key)\n",
        "  elif(ch == 2):\n",
        "      c = int(input(\"Delete OLD -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter Non-productive Keyword to delete: \")\n",
        "        non_productive_keywords.remove(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          print(f\"\\nCurrent List: {non_productive_keywords}\")\n",
        "          key = input(\"Enter keyword to remove (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          try:\n",
        "            non_productive_keywords.remove(key)\n",
        "          except ValueError:\n",
        "            print(f\"'{key}' not found in the list. Try again.\")\n",
        "        print(\"\\nFinal List:\", non_productive_keywords)\n",
        "  elif(ch == 3):\n",
        "      print(\"All Non-productive Keywords: \", \", \".join(non_productive_keywords))\n",
        "\n"
      ],
      "metadata": {
        "id": "kbmy2FiGqLsg"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Enter or delete pre-saved Domains\n",
        "\n",
        "def domain_management(prod_domains, nprod_domains):\n",
        "  choice = int(input(\"Enter choice to manage domains (1 or 2) : 1. Productive  2. Non-productive  3. Display All -> \"))\n",
        "  if(choice == 1):\n",
        "    get_productive_domain(prod_domains)\n",
        "  elif(choice == 2):\n",
        "    get_nonproductive_domain(nprod_domains)\n",
        "  elif choice not in [1, 2]:\n",
        "    print(\"Displaying All....\")\n",
        "    print(\"All Productive Domains:\", \", \".join(prod_domains))\n",
        "    print(\"All Non-productive Domains:\", \", \".join(nprod_domains))\n",
        "  return prod_domains, nprod_domains\n",
        "\n",
        "\n",
        "def get_productive_domain(prod_domains):\n",
        "  ch = int(input(\"Enter choice to manage productive domains (1 or 2) : 1. Add new  2. Delete old  3. Display all->\"))\n",
        "  if(ch == 1):\n",
        "      c = int(input(\"Add NEW -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter new Productive Domain: \")\n",
        "        prod_domains.append(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          key = input(\"\\nEnter Domain to add (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          prod_domains.append(key)\n",
        "  elif(ch == 2):\n",
        "      c = int(input(\"Delete OLD -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter Productive Domain to delete: \")\n",
        "        prod_domains.remove(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          print(f\"\\nCurrent List: {prod_domains}\")\n",
        "          key = input(\"Enter domain to remove (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          try:\n",
        "            prod_domains.remove(key)\n",
        "          except ValueError:\n",
        "            print(f\"'{key}' not found in the list. Try again.\")\n",
        "        print(\"\\nFinal List: \", prod_domains)\n",
        "  elif(ch == 3):\n",
        "      print(\"All Productive Domains: \", \", \".join(prod_domains))\n",
        "\n",
        "\n",
        "def get_nonproductive_domain(nprod_domains):\n",
        "  ch = int(input(\"Enter choice to manage non-productive domains (1 or 2) : 1. Add new  2. Delete old  3. Display all->\"))\n",
        "  if(ch == 1):\n",
        "      c = int(input(\"Add NEW -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter new Non-productive Domain: \")\n",
        "        nprod_domains.append(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          key = input(\"\\nEnter domain to add (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          nprod_domains.append(key)\n",
        "  elif(ch == 2):\n",
        "      c = int(input(\"Delete OLD -> Operation mode: 1. Single  2. Bulk : \"))\n",
        "      if(c == 1):\n",
        "        key = input(\"Enter Non-productive Domain to delete: \")\n",
        "        nprod_domains.remove(key)\n",
        "      if(c == 2):\n",
        "        while True:\n",
        "          print(f\"\\nCurrent List: {nprod_domains}\")\n",
        "          key = input(\"Enter domain to remove (or 'done' to finish): \")\n",
        "          if key.lower() == \"done\":\n",
        "            break\n",
        "          try:\n",
        "            nprod_domains.remove(key)\n",
        "          except ValueError:\n",
        "            print(f\"'{key}' not found in the list. Try again.\")\n",
        "        print(\"\\nFinal List:\", nprod_domains)\n",
        "  elif(ch == 3):\n",
        "      print(\"All Non-productive Domain: \", \", \".join(nprod_domains))\n"
      ],
      "metadata": {
        "id": "cUQTanwdXa_W"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Function to get metadata from URL\n",
        "def get_metadata_from_url(url):\n",
        "    try:\n",
        "        headers = {'User-Agent': choice(user_agents)}\n",
        "        response = requests.get(url, headers=headers)\n",
        "        response.raise_for_status()\n",
        "        soup = BeautifulSoup(response.text, 'html.parser')\n",
        "\n",
        "        title = soup.find('title').get_text() if soup.find('title') else 'No title'\n",
        "\n",
        "        description = soup.find('meta', attrs={'name': 'description'})\n",
        "        if description:\n",
        "            description = description.get('content')\n",
        "        else:\n",
        "            description = soup.find('meta', attrs={'property': 'og:description'})\n",
        "            description = description.get('content') if description else 'No description'\n",
        "\n",
        "        image = soup.find('meta', attrs={'property': 'og:image'})\n",
        "        if image:\n",
        "            image = image.get('content')\n",
        "        else:\n",
        "            image = soup.find('link', attrs={'rel': 'image_src'})\n",
        "            image = image.get('href') if image else 'No image'\n",
        "\n",
        "        text = ' '.join(p.get_text() for p in soup.find_all('p'))\n",
        "\n",
        "        return {\n",
        "            \"title\": title,\n",
        "            \"description\": description,\n",
        "            \"image\": image,\n",
        "            \"url\": url,\n",
        "            \"text\": text\n",
        "        }\n",
        "    except requests.RequestException as e:\n",
        "        return {\n",
        "            \"title\": \"Error\",\n",
        "            \"description\": str(e),\n",
        "            \"image\": \"No image\",\n",
        "            \"url\": url,\n",
        "            \"text\": \"\"\n",
        "        }\n",
        "\n",
        "# Preprocess text function\n",
        "def preprocess_text(text):\n",
        "    if not text:\n",
        "        return \"\"\n",
        "    text = text.lower()\n",
        "    text = ''.join([char for char in text if char not in string.punctuation])\n",
        "    tokens = text.split()\n",
        "    tokens = [word for word in tokens if word not in stop_words]\n",
        "    return ' '.join(tokens)\n",
        "\n",
        "# Translate text if not in English\n",
        "def translate_text_if_needed(text):\n",
        "    try:\n",
        "        language = detect(text)\n",
        "        if language != 'en':\n",
        "            translated = translator.translate(text, dest='en')\n",
        "            return translated.text\n",
        "        return text\n",
        "    except Exception as e:\n",
        "        return text\n",
        "\n",
        "# Preprocess metadata function\n",
        "def preprocess_metadata(metadata):\n",
        "    title = translate_text_if_needed(metadata.get('title', ''))\n",
        "    description = translate_text_if_needed(metadata.get('description', ''))\n",
        "    text = translate_text_if_needed(metadata.get('text', ''))\n",
        "    clean_title = preprocess_text(title)\n",
        "    clean_description = preprocess_text(description)\n",
        "    clean_text = preprocess_text(text)\n",
        "    combined_clean_content = f\"{clean_title} {clean_description} {clean_text}\".strip()\n",
        "    return combined_clean_content\n",
        "\n",
        "# Function to extract domain from URL\n",
        "def extract_domain(url):\n",
        "    return url.split('//')[-1].split('/')[0]\n",
        "\n",
        "# Keyword matching function\n",
        "def count_keywords(text, keywords):\n",
        "    tokens = text.split()\n",
        "    return sum(token in keywords for token in tokens)\n",
        "\n",
        "# Function to create feature matrix\n",
        "def create_feature_matrix(df):\n",
        "    tfidf_matrix = vectorizer.fit_transform(df['clean_content'])\n",
        "\n",
        "    keyword_counts = df[['productive_keyword_count', 'non_productive_keyword_count']].values\n",
        "    return np.hstack((tfidf_matrix.toarray(), keyword_counts))"
      ],
      "metadata": {
        "id": "2FgEXFEYcPls"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def prepare_dataset(productive_keywords, non_productive_keywords, dataset_url):\n",
        "  data = pd.read_csv(dataset_url)\n",
        "  data.drop(data.columns.difference(['url', 'label']), axis=1, inplace=True)\n",
        "  #Extract domain\n",
        "  data['domain'] = data['url'].apply(extract_domain)\n",
        "  # Scrape Content and Metadata\n",
        "  data['metadata'] = data['url'].apply(get_metadata_from_url)\n",
        "  # Apply Preprocessing to Metadata\n",
        "  data['clean_content'] = data['metadata'].apply(preprocess_metadata)\n",
        "  data['productive_keyword_count'] = data['clean_content'].apply(lambda x: count_keywords(x, productive_keywords))\n",
        "  data['non_productive_keyword_count'] = data['clean_content'].apply(lambda x: count_keywords(x, non_productive_keywords))\n",
        "  return data"
      ],
      "metadata": {
        "id": "InJ1KuVRcg6F"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def train_test_model(data):\n",
        "  X = create_feature_matrix(data)\n",
        "  y = data['label']\n",
        "\n",
        "  # Train/Test Split\n",
        "  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)\n",
        "\n",
        "  # Train Model\n",
        "  model = MultinomialNB()\n",
        "  model.fit(X_train, y_train)\n",
        "\n",
        "  # Evaluate Model\n",
        "  y_pred = model.predict(X_test)\n",
        "  print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
        "  #print(classification_report(y_test, y_pred))\n",
        "  #print(classification_report(y_test, y_pred, target_names=['Productive', 'Non-productive', 'Neutral']))\n",
        "  return model"
      ],
      "metadata": {
        "id": "qx6SC5UgpW9C"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def classify_url(productive_keywords, non_productive_keywords, model, url):\n",
        "    metadata = get_metadata_from_url(url)\n",
        "    clean_content = preprocess_metadata(metadata)\n",
        "    vectorized_content = vectorizer.transform([clean_content])\n",
        "    keyword_counts = np.array([[\n",
        "        count_keywords(clean_content, productive_keywords),\n",
        "        count_keywords(clean_content, non_productive_keywords)\n",
        "    ]])\n",
        "    feature_vector = np.hstack((vectorized_content.toarray(), keyword_counts))\n",
        "    prediction = model.predict(feature_vector)[0]\n",
        "    return prediction"
      ],
      "metadata": {
        "id": "NiSBTQyFqBAT"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def set_push(dataset_url, url, prediction, flag):\n",
        "  data = pd.read_csv(dataset_url)\n",
        "  if flag == 1:\n",
        "    pred = int(input(\"Enter Prediction to enlist: 0. Neutral  1. Productive  2. Non-productive : \"))\n",
        "    if pred == 2:\n",
        "      prediction = \"Non-productive\"\n",
        "    elif pred == 0:\n",
        "      prediction = \"Neutral\"\n",
        "    elif pred == 1:\n",
        "      prediction = \"Productive\"\n",
        "\n",
        "  new_row = {'index': len(data) + 1, 'url': url, 'label': prediction}\n",
        "  new_row_df = pd.DataFrame([new_row])\n",
        "  data = pd.concat([data, new_row_df], ignore_index=True)\n",
        "  data.to_csv(dataset_url, index=False)\n",
        "  print(\"Row added into Dataset\")\n",
        "  return data"
      ],
      "metadata": {
        "id": "JfG0ejnG6QOz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def add_rows(dataset_url):\n",
        "    data = pd.read_csv(dataset_url)\n",
        "    url = input(\"Enter url to add: \")\n",
        "    prediction = input(\"Enter prediction (Productive or Non-productive or Neutral): \")\n",
        "    new_row = {'index': len(data) + 1, 'url': url, 'label': prediction}\n",
        "    new_row_df = pd.DataFrame([new_row])\n",
        "    data = pd.concat([data, new_row_df], ignore_index=True)\n",
        "    data.to_csv(dataset_url, index=False)\n",
        "    print(\"Row added into Dataset\")\n",
        "    return data"
      ],
      "metadata": {
        "id": "Way5ylblAmmk"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def remove_rows(dataset_url, a, b):\n",
        "    data = pd.read_csv(dataset_url)\n",
        "    # Remove rows between indices a and b\n",
        "    data.drop(data.index[a-1:b], inplace=True)\n",
        "    # Save the updated DataFrame back to the same CSV file\n",
        "    data.to_csv(dataset_url, index=False)\n",
        "    print(f\"Rows {a} to {b} removed from Dataset\")\n",
        "    return data"
      ],
      "metadata": {
        "id": "HIiLOCwO-WpP"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def display_dataset(dataset_url):\n",
        "  data = pd.read_csv(dataset_url)\n",
        "  display(data.tail())"
      ],
      "metadata": {
        "id": "_mE1CNFG7LOz"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def final_execute(productive_keywords, non_productive_keywords):\n",
        "  print(\"All Productive Keywords:\", \", \".join(productive_keywords))\n",
        "  print(\"All Non-productive Keywords:\", \", \".join(non_productive_keywords))\n",
        "  print(\"All Productive Domains:\", \", \".join(prod_domains))\n",
        "  print(\"All Non-productive Domains:\", \", \".join(nprod_domains))\n",
        "  dataset_url = '/content/drive/MyDrive/Sayan RP files/Datasets/URL_Dataset(Sheet1).csv'\n",
        "\n",
        "  while True:\n",
        "    option = input(\"Do you want to modify? Y or N : \")\n",
        "    if ((option == 'Y') or (option == 'y')):\n",
        "      choice = input(\"Do you want to modify keywords or domains? Enter 'K' for keywords or 'D' for domains: \")\n",
        "      if choice.lower() == 'k':\n",
        "        productive_keywords, non_productive_keywords = keyword_management(productive_keywords, non_productive_keywords)\n",
        "      elif choice.lower() == 'd':\n",
        "        prod_domains, nprod_domains = domain_management(prod_domains, nprod_domains)\n",
        "      else:\n",
        "        print(\"Invalid choice. Please enter 'K' or 'D'.\")\n",
        "        continue\n",
        "    else:\n",
        "        print(\"All Productive Keywords:\", \", \".join(productive_keywords))\n",
        "        print(\"All Non-productive Keywords:\", \", \".join(non_productive_keywords))\n",
        "        print(\"All Productive Domains:\", \", \".join(prod_domains))\n",
        "        print(\"All Non-productive Domains:\", \", \".join(nprod_domains))\n",
        "        break\n",
        "\n",
        "  prod = productive_keywords\n",
        "  non_prod = non_productive_keywords\n",
        "  pdom = prod_domains\n",
        "  ndom = nprod_domains\n",
        "\n",
        "  data = prepare_dataset(prod, non_prod, dataset_url)\n",
        "  model = train_test_model(data)\n",
        "\n",
        "  new_url = input(\"Enter url to check for Productive, Non-productive, or Neutral: \")\n",
        "  # Extract domain and check against prod_domains and nprod_domains\n",
        "  domain = extract_domain(new_url)\n",
        "  if domain in prod_domains:\n",
        "    classification = 'Productive'\n",
        "  elif domain in nprod_domains:\n",
        "    classification = 'Non-productive'\n",
        "  else:\n",
        "    classification = classify_url(prod, non_prod, model, new_url)\n",
        "\n",
        "  print(f\"The URL {new_url} is classified as {classification}\")\n",
        "\n",
        "  flag = input(\"Satisfied with the output? Y or N : \")\n",
        "  if ((flag == 'Y') or (flag == 'y')):\n",
        "    push = input('Do you want to add the results to the Dataset? Y or N : ')\n",
        "    if ((push == 'Y') or (push == 'y')):\n",
        "      data = set_push(dataset_url, new_url, classification, 0)\n",
        "      display_dataset(dataset_url)\n",
        "    else:\n",
        "      print(\"Not Added to the Dataset\")\n",
        "      display_dataset(dataset_url)\n",
        "  else:\n",
        "    push = input('Do you want to add corrected results to the Dataset? Y or N : ')\n",
        "    if ((push == 'Y') or (push == 'y')):\n",
        "      data = set_push(dataset_url, new_url, classification, 1)\n",
        "      display_dataset(dataset_url)\n",
        "    else:\n",
        "      print(\"Not Added to the Dataset\")\n",
        "      display_dataset(dataset_url)\n"
      ],
      "metadata": {
        "id": "gVrjuyUf5pyr"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def main():\n",
        "  final_execute(productive_keywords, non_productive_keywords)"
      ],
      "metadata": {
        "id": "KWLOYgfKcfxj"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "if __name__==\"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "id": "hL_21oeGSlKb"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}