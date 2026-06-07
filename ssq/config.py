from pathlib import Path

# 双色球规则
RED_MAX = 33
BLUE_MAX = 16
RED_COUNT = 6

# 路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "ssq.db"

# 数据源 - 中国福彩官方API
DATA_URL = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"
DATA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Referer": "https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/",
    "Accept": "application/json",
}

# 抓取参数
PAGE_SIZE = 100
REQUEST_TIMEOUT = 15
RETRY_COUNT = 3
RETRY_DELAY = 2
