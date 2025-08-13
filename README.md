# AI Career Path Predictor + Skill Gap Analyzer

## 📌 Overview
This project uses the **Adzuna Job Search API** to fetch real-time job postings, analyze skills required in the market, and compare them with a user’s current skills.  
The aim is to **predict a suitable career path** and **identify skill gaps** so users can take targeted action to improve their job prospects.

## ✨ Features
- 🔍 **Fetch real-time job postings** from Adzuna API
- 📊 **Analyze in-demand skills** from live data
- 🧠 **AI-based career suggestions** (coming soon)
- 📂 Save results in **JSON or CSV**
- 🌐 Simple **Streamlit UI** for viewing data

## 🛠 Tech Stack
- **Python 3.10+**
- **Adzuna API**
- **Requests** (for API calls)
- **Pandas** (for data handling)
- **Streamlit** (for web UI)

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ai-career-path-predictor.git
cd ai-career-path-predictor
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate the virtual environment**

**Windows**
```bash
venv\Scripts\activate
```

**Mac/Linux**
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🔑 Get Adzuna API Credentials
- Go to [Adzuna API](https://developer.adzuna.com/)
- Sign up and get your **App ID** and **App Key**
- Replace them in the script:
```python
APP_ID = "your_app_id_here"
APP_KEY = "your_app_key_here"
```

## ▶️ Usage

**Run the scraper**
```bash
python adzuna_scraper.py
```

**Run the Streamlit app**
```bash
streamlit run adzuna_scraper.py
```
Your browser will open automatically with the job listings.

## 📁 Output
- **adzuna_jobs.json** → Raw API data in JSON format  
- **adzuna_jobs.csv** → Clean table of job listings  

Both contain:
- Job title
- Company
- Location
- Salary
- Job description
- URL to apply

## 🛣 Roadmap
- [ ] Add AI skill gap analysis
- [ ] Recommend learning resources
- [ ] Predict career transitions
- [ ] Add LinkedIn profile import

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## 📜 License
My License

---

💡 Made with ❤️ to help job seekers find their path faster.
