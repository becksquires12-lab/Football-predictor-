"""Tests for the FBRef web scraper."""

import pytest

from src.data.scraper_fbref import FBRefScraper


class TestFBRefScraper:
    def test_league_fbref_id_mapping(self):
        assert FBRefScraper._league_fbref_id("EPL") == 9
        assert FBRefScraper._league_fbref_id("LA_LIGA") == 12
        assert FBRefScraper._league_fbref_id("BUNDESLIGA") == 20
        assert FBRefScraper._league_fbref_id("SERIE_A") == 11
        assert FBRefScraper._league_fbref_id("LIGUE_1") == 13
        # Unknown league returns default
        assert FBRefScraper._league_fbref_id("UNKNOWN") == 9

    def test_get_stat_pair_valid(self):
        from bs4 import BeautifulSoup
        html = "<div>5</div><div>Corners</div><div>7</div>"
        soup = BeautifulSoup(html, "html.parser")
        divs = soup.find_all("div")

        result = FBRefScraper._get_stat_pair(divs, 1)
        assert result == (5, 7)

    def test_get_stat_pair_invalid(self):
        from bs4 import BeautifulSoup
        html = "<div>N/A</div><div>Corners</div><div>7</div>"
        soup = BeautifulSoup(html, "html.parser")
        divs = soup.find_all("div")

        result = FBRefScraper._get_stat_pair(divs, 1)
        assert result is None

    def test_parse_empty_scores_page(self):
        scraper = FBRefScraper()
        result = scraper._parse_scores_page("<html><body></body></html>", "EPL")
        assert result == []
