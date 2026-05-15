"""
Download papers with proxy support + more sources
"""
import os
import sys
import requests

PAPERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "papers")
os.makedirs(PAPERS_DIR, exist_ok=True)

# Read proxy from env var (e.g. HTTP_PROXY=http://127.0.0.1:7890)
_proxy_url = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
PROXY = {"http": _proxy_url, "https": _proxy_url} if _proxy_url else None

PAPERS = [
    # Already downloaded, skip
    # {"name": "NBER_w31925_AI_CrossBorder_Trade.pdf", ...},

    # Try alternate OECD link
    {
        "name": "OECD_GVC_Development_Report_2023.pdf",
        "url": "https://www.wto.org/english/res_e/booksp_e/gvc_dev_rep_2023_e.pdf",
        "title": "Global Value Chain Development Report 2023 (WTO/ADB/OECD)",
    },
    # NBER: global value chains
    {
        "name": "NBER_w26580_GVC_Gravity.pdf",
        "url": "https://www.nber.org/system/files/working_papers/w26580/w26580.pdf",
        "title": "Global Value Chains and Trade Elasticities (Antras et al., NBER 26580)",
    },
    # Open access from Journal of Economic Structures
    {
        "name": "LMDI_IO_Energy_Emissions_China_2022.pdf",
        "url": "https://link.springer.com/content/pdf/10.1186/s40008-022-00270-w.pdf",
        "title": "LMDI Input-Output analysis of energy-related CO2 emissions in China (2022, Open Access)",
    },
    # IMF Working Paper (free)
    {
        "name": "IMF_GVC_Bottleneck_Imports_2023.pdf",
        "url": "https://www.imf.org/-/media/Files/Publications/WP/2023/English/wpiea2023050-print-pdf.ashx",
        "title": "Global Value Chain Disruptions and Inflation (IMF WP 2023)",
    },
    # WTO report
    {
        "name": "WTO_World_Trade_Report_2023_Re-globalization.pdf",
        "url": "https://www.wto.org/english/res_e/booksp_e/wtr23_e/wtr23_e.pdf",
        "title": "WTO World Trade Report 2023: Re-globalization",
    },
]

def download_paper(paper):
    filepath = os.path.join(PAPERS_DIR, paper["name"])
    if os.path.exists(filepath):
        print(f"  SKIP: {paper['name']}")
        return True
    try:
        resp = requests.get(paper["url"], timeout=60, proxies=PROXY)
        if resp.status_code == 200:
            with open(filepath, "wb") as f:
                f.write(resp.content)
            kb = os.path.getsize(filepath) / 1024
            print(f"  OK: {paper['name']} ({kb:.0f} KB)")
            return True
        else:
            print(f"  FAIL HTTP {resp.status_code}: {paper['name']}")
            return False
    except Exception as e:
        print(f"  FAIL: {paper['name']} - {str(e)[:80]}")
        return False

if __name__ == "__main__":
    print(f"Downloading to: {PAPERS_DIR}\n")
    ok = 0
    for p in PAPERS:
        print(f"  {p['title'][:80]}...")
        if download_paper(p):
            ok += 1
    print(f"\nResult: {ok}/{len(PAPERS)} downloaded")
