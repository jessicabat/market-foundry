import os
import sys
from pathlib import Path
from newspaper import Article, Config

# Add src to path so we can import safe_compact
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src/knn_pipeline'))
from mf_utils.cleaning import safe_compact

REF_DIR = Path("reference_docs_clean/news_article")
REF_DIR.mkdir(parents=True, exist_ok=True)

NEWS_URLS = [
    "https://www.reuters.com/business/nvidia-close-finalizing-30-billion-investment-openai-funding-round-ft-reports-2026-02-20/",
    "https://www.marketwatch.com/story/prediction-market-etfs-may-open-the-door-to-all-sorts-of-wall-street-tomfoolery-df17805f?mod=home_lead",
    "https://finance.yahoo.com/news/jpmorgan-concedes-closed-trumps-accounts-205827389.html",
    "https://apnews.com/article/stocks-markets-iran-ai-trump-9105f5156c294507738154a714dcd13d",
    "https://www.reuters.com/legal/government/ecb-fines-jpmorgan-122-mln-euros-misreporting-capital-requirements-2026-02-19/",
    "https://www.reuters.com/business/bofa-commits-25-billion-private-credit-deals-bloomberg-news-reports-2026-02-19/",
    "https://apnews.com/article/nyse-tokenization-trading-platform-588b84ea6d3b4745da42d58bb80bd718",
    "https://apnews.com/article/asia-nvidia-earnings-us-stocks-71372f3476dd13c33d316819bf902b17",
    "https://finance.yahoo.com/news/affirm-holdings-inc-afrm-announces-110334373.html",
    "https://finance.yahoo.com/news/uber-invest-over-100-million-120148489.html",
    "https://finance.yahoo.com/news/softbank-swings-profit-valuation-boost-074014227.html",
    "https://finance.yahoo.com/news/robinhood-cfo-defends-its-future-as-shares-slide-9-after-earnings-184625919.html",
    "https://finance.yahoo.com/news/google-hit-fresh-eu-antitrust-173215204.html",
    "https://www.reuters.com/business/retail-consumer/saks-global-collapse-shows-struggles-department-store-model-kering-ceo-says-2026-02-10/",
    "https://seekingalpha.com/news/4554910-key-deals-this-week-visa-hims-hers-health-salesforce-and-more",
    "https://seekingalpha.com/news/4554814-cybersecurity-stocks-fall-after-anthropic-unveils-claude-code-security",
    "https://seekingalpha.com/news/4553542-figma-surges-after-q4-revenue-jumps-40-2026-full-year-guidance-shows-38-revenue-jump",
    "https://seekingalpha.com/news/4553605-ebay-to-acquire-depop-from-etsy-for-12b-in-cash",
    "https://www.forbes.com/sites/aliciapark/2026/02/10/vcs-are-throwing-money-at-recent-college-grads-to-build-prediction-markets/",
    "https://www.forbes.com/sites/kellyphillipserb/2026/01/26/treasury-cancels-all-booz-allen-contracts-over-leak-of-billionaires-tax-data/",
    "https://www.businessinsider.com/centerview-partners-settles-junior-banker-case-before-trial-2026-2",
    "https://www.businessinsider.com/pwc-engineers-launch-ai-agent-enterprise-grade-spreadsheets-big-four-2026-2",
    "https://www.businessinsider.com/short-seller-campaigns-number-increased-2025-tech-ai-stocks-2026-2",
    "https://www.reuters.com/legal/transactional/txnm-energy-gets-ferc-approval-115-billion-blackstone-deal-2026-02-20/"
]

def main():
    total_urls = len(NEWS_URLS)
    print(f"🌐 Attempting to scrape {total_urls} news articles...")
    
    # Configure the scraper to look like a real Mac web browser
    # This prevents sites from immediately blocking the script with a 403 error
    config = Config()
    config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0'
    config.request_timeout = 10
    
    success_count = 0
    failed_count = 0
    
    for i, url in enumerate(NEWS_URLS, start=1):
        try:
            print(f"[{i}/{total_urls}] Downloading: {url.split('.com')[0] + '.com/...'}")
            article = Article(url, config=config)
            article.download()
            article.parse()
            
            raw_text = article.title + "\n\n" + article.text
            clean_text = safe_compact(raw_text)
            
            # Guardrail: If it's less than 200 characters, we likely hit a paywall popup
            if len(clean_text) < 200:
                print(f"   ⚠️ Warning: Extracted text is too short ({len(clean_text)} chars). Likely a paywall. Skipping.")
                failed_count += 1
                continue
            
            safe_title = "".join([c if c.isalnum() else "_" for c in article.title])[:50]
            out_filepath = REF_DIR / f"news_{i}_{safe_title}.txt"
            
            with open(out_filepath, "w", encoding="utf-8") as f:
                f.write(clean_text)
                
            print(f"   ✅ Saved: {out_filepath.name}")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to scrape: {e}")
            failed_count += 1

    print("\n" + "="*40)
    print("📊 SCRAPING SUMMARY")
    print("="*40)
    print(f"Total Attempted : {total_urls}")
    print(f"Successfully added: {success_count}")
    print(f"Failed / Blocked  : {failed_count}")
    print("="*40)

if __name__ == "__main__":
    main()