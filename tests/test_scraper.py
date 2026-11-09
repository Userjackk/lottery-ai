"""测试抓取异常处理"""
import pytest
from ssq.scraper import fetch_page
from unittest.mock import patch, MagicMock


def test_fetch_page_retries_and_raises():
    """连续失败应抛出带中文提示的 RuntimeError"""
    with patch("ssq.scraper.requests.get") as mock_get:
        mock_get.side_effect = Exception("network error")
        with pytest.raises(RuntimeError) as exc_info:
            fetch_page(1, 10)
        assert "抓取数据失败" in str(exc_info.value)
        # 应重试 RETRY_COUNT 次
        assert mock_get.call_count == 3


def test_fetch_page_returns_data():
    """正常响应应解析数据"""
    with patch("ssq.scraper.requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "result": [{
                "code": "2026060",
                "date": "2026-09-20T00:00:00+08:00",
                "red": "03,12,15,19,22,30",
                "blue": "08",
            }]
        }
        mock_get.return_value = mock_resp
        draws = fetch_page(1, 10)
        assert len(draws) == 1
        assert draws[0]["draw_no"] == "2026060"
