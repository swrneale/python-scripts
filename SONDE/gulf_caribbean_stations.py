"""
gulf_caribbean_stations.py
==========================
Radiosonde / upper-air sounding station metadata for the
Gulf of Mexico – Caribbean – Central America region
(lat 8-38 N, lon 105-60 W).

Sources
-------
- IGRA v2 station list (primary): https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt
  (downloaded / verified April 2026 – each station's IGRA2 ID, lat, lon, elev,
  and period of record taken directly from that file)
- BADC/CEDA North & Central America station list:
  https://artefacts.ceda.ac.uk/badc_datadocs/radiosglobe/northamerica.html
- NWS Upper-Air Network information: https://www.weather.gov/upperair/net-info
- University of Wyoming Radiosonde Archive: https://weather.uwyo.edu/upperair/sounding.shtml

Fields
------
wmo_id  : 5-digit WMO station number (string)
lat     : latitude, decimal degrees North
lon     : longitude, decimal degrees East (negative = West)
elev    : elevation above mean sea level, metres
country : country name
active  : approximate last year of record in IGRA v2 (as of April 2026)
notes   : optional remarks on station status / name changes

Stations are grouped by subregion for readability, but the dictionary
key is a short, unique station name (no spaces) suitable as a file-system
label or plot label.

All coordinates and elevations are taken from the IGRA v2 station list
unless noted otherwise.
"""

STATION_NAMES = {

    # ------------------------------------------------------------------
    # US Gulf Coast – Texas (active NWS network unless noted)
    # ------------------------------------------------------------------
    'Brownsville': {
        'wmo_id': '72250', 'lat': 25.92, 'lon': -97.42, 'elev':  7,
        'country': 'USA', 'active': 2026,
        'notes': 'Brownsville/Intl, TX; NWS launch site',
    },
    'CorpusChristi': {
        'wmo_id': '72251', 'lat': 27.78, 'lon': -97.51, 'elev': 15,
        'country': 'USA', 'active': 2026,
        'notes': 'Corpus Christi/Intl, TX; NWS launch site',
    },
    'FortWorth': {
        'wmo_id': '72249', 'lat': 32.84, 'lon': -97.30, 'elev': 195,
        'country': 'USA', 'active': 2026,
        'notes': 'Fort Worth, TX (NWS Dallas-Ft Worth); NWS launch site',
    },
    'DelRio': {
        'wmo_id': '72261', 'lat': 29.37, 'lon': -100.92, 'elev': 314,
        'country': 'USA', 'active': 2026,
        'notes': 'Del Rio/Intl, TX; NWS launch site',
    },
    'Midland': {
        'wmo_id': '72265', 'lat': 31.94, 'lon': -102.19, 'elev': 875,
        'country': 'USA', 'active': 2026,
        'notes': 'Midland Regional Airterm, TX; NWS launch site',
    },
    'Amarillo': {
        'wmo_id': '72363', 'lat': 35.23, 'lon': -101.71, 'elev': 1095,
        'country': 'USA', 'active': 2026,
        'notes': 'Amarillo/Intl, TX; NWS launch site',
    },
    'Houston': {
        'wmo_id': '72243', 'lat': 29.60, 'lon': -95.17, 'elev': 15,
        'country': 'USA', 'active': 2026,
        'notes': 'Houston/Ellington Field NAS, TX; NWS launch site',
    },
    'Victoria': {
        'wmo_id': '72244', 'lat': 28.85, 'lon': -96.92, 'elev': 34,
        'country': 'USA', 'active': 1999,
        'notes': 'Victoria Regional, TX; discontinued',
    },

    # ------------------------------------------------------------------
    # US Gulf Coast – Louisiana / Mississippi / Alabama
    # ------------------------------------------------------------------
    'LakeCharles': {
        'wmo_id': '72240', 'lat': 30.13, 'lon': -93.22, 'elev':  5,
        'country': 'USA', 'active': 2026,
        'notes': 'Lake Charles/Mun., LA; NWS launch site',
    },
    'Slidell': {
        'wmo_id': '72233', 'lat': 30.34, 'lon': -89.83, 'elev': 10,
        'country': 'USA', 'active': 2026,
        'notes': 'Slidell/Mun. LA (replaces New Orleans); NWS launch site',
    },
    'Shreveport': {
        'wmo_id': '72248', 'lat': 32.45, 'lon': -93.84, 'elev': 85,
        'country': 'USA', 'active': 2026,
        'notes': 'Shreveport Regional, LA; NWS launch site',
    },
    'Jackson': {
        'wmo_id': '72235', 'lat': 32.32, 'lon': -90.08, 'elev': 91,
        'country': 'USA', 'active': 2026,
        'notes': 'Jackson/Allen C. Thompson Field, MS; NWS launch site',
    },
    'Birmingham': {
        'wmo_id': '72230', 'lat': 33.18, 'lon': -86.78, 'elev': 174,
        'country': 'USA', 'active': 2026,
        'notes': 'Birmingham, AL; NWS launch site (replaced older 72228)',
    },

    # Historical / discontinued Gulf Coast US
    'NewOrleans': {
        'wmo_id': '72231', 'lat': 29.98, 'lon': -90.25, 'elev':  3,
        'country': 'USA', 'active': 1950,
        'notes': 'New Orleans, LA; historic site, replaced by Slidell 72233',
    },
    'Boothville': {
        'wmo_id': '72232', 'lat': 29.33, 'lon': -89.40, 'elev':  0,
        'country': 'USA', 'active': 1988,
        'notes': 'Boothville WSCMO, LA; discontinued',
    },
    'Apalachicola': {
        'wmo_id': '72220', 'lat': 29.73, 'lon': -85.03, 'elev':  6,
        'country': 'USA', 'active': 1991,
        'notes': 'Apalachicola Muni, FL; discontinued',
    },

    # ------------------------------------------------------------------
    # US Gulf Coast / Southeast – Florida
    # ------------------------------------------------------------------
    'Valparaiso': {
        'wmo_id': '72221', 'lat': 30.48, 'lon': -86.52, 'elev': 29,
        'country': 'USA', 'active': 2026,
        'notes': 'Valparaiso/Eglin AFB, FL; NWS launch site',
    },
    'Tallahassee': {
        'wmo_id': '72214', 'lat': 30.45, 'lon': -84.30, 'elev': 53,
        'country': 'USA', 'active': 2024,
        'notes': 'Tallahassee/Mun., FL; NWS launch site',
    },
    'Jacksonville': {
        'wmo_id': '72206', 'lat': 30.48, 'lon': -81.70, 'elev': 10,
        'country': 'USA', 'active': 2026,
        'notes': 'Jacksonville/Intl., FL; NWS launch site',
    },
    'Tampa': {
        'wmo_id': '72210', 'lat': 27.71, 'lon': -82.40, 'elev': 13,
        'country': 'USA', 'active': 2026,
        'notes': 'Tampa Bay Area, FL; NWS launch site',
    },
    'CapeKennedy': {
        'wmo_id': '74794', 'lat': 28.47, 'lon': -80.55, 'elev':  3,
        'country': 'USA', 'active': 2026,
        'notes': 'Cape Kennedy / KSC, FL; special-purpose launch site (NASA/USAF)',
    },
    'Miami': {
        'wmo_id': '72202', 'lat': 25.75, 'lon': -80.38, 'elev':  4,
        'country': 'USA', 'active': 2026,
        'notes': 'Miami Intl Airport, FL; NWS launch site',
    },
    'KeyWest': {
        'wmo_id': '72201', 'lat': 24.55, 'lon': -81.79, 'elev': 13,
        'country': 'USA', 'active': 2026,
        'notes': 'Key West/Intl., FL; NWS launch site',
    },
    'WestPalmBeach': {
        'wmo_id': '72203', 'lat': 26.68, 'lon': -80.10, 'elev':  7,
        'country': 'USA', 'active': 1995,
        'notes': 'West Palm Beach Intl, FL; discontinued',
    },

    # ------------------------------------------------------------------
    # Mexico – Gulf coast, Yucatan, Pacific near Gulf
    # ------------------------------------------------------------------
    'Monterrey': {
        'wmo_id': '76394', 'lat': 25.87, 'lon': -100.23, 'elev': 448,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Aerop. Internacional Monterrey, NL',
    },
    'Matamoros': {
        'wmo_id': '76393', 'lat': 25.77, 'lon': -97.52, 'elev': 10,
        'country': 'Mexico', 'active': 1995,
        'notes': 'Matamoros, TAM; Gulf coast just south of Brownsville',
    },
    'Tuxpan': {
        'wmo_id': '76595', 'lat': 20.95, 'lon': -97.40, 'elev': 10,
        'country': 'Mexico', 'active': 1999,
        'notes': 'Tuxpan, VER; Gulf coast',
    },
    'Coatzacoalcos': {
        'wmo_id': '76750', 'lat': 18.13, 'lon': -94.40, 'elev': 10,
        'country': 'Mexico', 'active': 1999,
        'notes': 'Coatzacoalcos, VER; southern Gulf coast',
    },
    'Campeche': {
        'wmo_id': '76645', 'lat': 19.83, 'lon': -90.53, 'elev': 7,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Aerop. Internacional Campeche; active',
    },
    'Chetumal': {
        'wmo_id': '76679', 'lat': 18.50, 'lon': -88.32, 'elev': 10,
        'country': 'Mexico', 'active': 2010,
        'notes': 'Chetumal, QR; Caribbean coast of Yucatan',
    },
    'Tampico': {
        'wmo_id': '76549', 'lat': 22.28, 'lon': -97.85, 'elev': 15,
        'country': 'Mexico', 'active': 2006,
        'notes': 'Tampico, TAM; limited record 2002-2006 in IGRA',
    },
    'CiudadVictoria': {
        'wmo_id': '76491', 'lat': 23.73, 'lon': -99.11, 'elev': 336,
        'country': 'Mexico', 'active': 1951,
        'notes': 'Ciudad Victoria, TAM; early historic record only',
    },
    'Zacatecas': {
        'wmo_id': '76526', 'lat': 22.75, 'lon': -102.51, 'elev': 2265,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Zacatecas Sondeo; active since 2000',
    },
    'Cancun': {
        'wmo_id': '76595', 'lat': 21.03, 'lon': -86.85, 'elev':  9,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Cancun, QR; active since 1995',
    },
    'Merida': {
        'wmo_id': '76644', 'lat': 20.95, 'lon': -89.65, 'elev': 10,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Aerop. Internacional Merida, YU; long record 1948-present',
    },
    'MexicoCity': {
        'wmo_id': '76679', 'lat': 19.40, 'lon': -99.20, 'elev': 2337,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Aerop. Internacional Mexico (Benito Juarez); long record 1948-present',
    },
    'Veracruz': {
        'wmo_id': '76692', 'lat': 19.14, 'lon': -96.11, 'elev': 11,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Hacienda Ylang Ylang / Veracruz, VER; long record 1952-present',
    },
    'CiudadDelCarmen': {
        'wmo_id': '76713', 'lat': 18.63, 'lon': -91.83, 'elev':  2,
        'country': 'Mexico', 'active': 2005,
        'notes': 'Ciudad del Carmen, CAM; very short record (2005 only in IGRA)',
    },
    'Villahermosa': {
        'wmo_id': '76743', 'lat': 17.98, 'lon': -92.92, 'elev':  6,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Villahermosa, TAB; active since 2005',
    },
    'Acapulco': {
        'wmo_id': '76805', 'lat': 16.76, 'lon': -99.75, 'elev':  3,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Acapulco, GRO; Pacific coast; active 1974-present',
    },
    'SalinaCruz': {
        'wmo_id': '76833', 'lat': 16.16, 'lon': -95.23, 'elev':  6,
        'country': 'Mexico', 'active': 2003,
        'notes': 'Salina Cruz, OAX; Pacific coast; limited record 2002-2003',
    },
    'Tapachula': {
        'wmo_id': '76903', 'lat': 14.89, 'lon': -92.30, 'elev': 114,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Tapachula, CHIS; near Guatemala border; active since 2011',
    },
    'Guadalajara': {
        'wmo_id': '76612', 'lat': 20.71, 'lon': -103.39, 'elev': 1554,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Guadalajara, JAL; Pacific side but commonly included; active 1979-present',
    },
    'Manzanillo': {
        'wmo_id': '76654', 'lat': 19.04, 'lon': -104.32, 'elev':  2,
        'country': 'Mexico', 'active': 2026,
        'notes': 'Manzanillo, COL; Pacific coast; active 1976-present',
    },
    'Torreon': {
        'wmo_id': '76382', 'lat': 25.53, 'lon': -103.45, 'elev': 1150,
        'country': 'Mexico', 'active': 2004,
        'notes': 'Torreon Airport; discontinued ~2004',
    },

    # ------------------------------------------------------------------
    # Cuba
    # ------------------------------------------------------------------
    'Havana': {
        'wmo_id': '78325', 'lat': 23.17, 'lon': -82.35, 'elev': 50,
        'country': 'Cuba', 'active': 1995,
        'notes': 'Casa Blanca / Havana; record ends 1995 in IGRA; Cuba does '
                 'not share data via GTS in recent decades',
    },
    'Camaguey': {
        'wmo_id': '78355', 'lat': 21.40, 'lon': -77.85, 'elev': 122,
        'country': 'Cuba', 'active': 1994,
        'notes': 'Camaguey; record ends 1994 in IGRA',
    },
    'GuantanamoBay': {
        'wmo_id': '78367', 'lat': 19.90, 'lon': -75.22, 'elev': 56,
        'country': 'Cuba', 'active': 1991,
        'notes': 'Guantanamo Bay NAS (US military); record ends 1991 in IGRA',
    },

    # ------------------------------------------------------------------
    # Jamaica
    # ------------------------------------------------------------------
    'Kingston': {
        'wmo_id': '78397', 'lat': 17.97, 'lon': -76.97, 'elev': 21,
        'country': 'Jamaica', 'active': 2026,
        'notes': 'Norman Manley / Windsor Airport, Kingston; active 1956-present',
    },

    # ------------------------------------------------------------------
    # Dominican Republic / Haiti
    # ------------------------------------------------------------------
    'SantoDomingo': {
        'wmo_id': '78486', 'lat': 18.47, 'lon': -69.87, 'elev': 14,
        'country': 'DominicanRepublic', 'active': 2026,
        'notes': 'Santo Domingo; active 1962-present',
    },
    'SabanaDelMar': {
        'wmo_id': '78467', 'lat': 19.05, 'lon': -69.38, 'elev': 11,
        'country': 'DominicanRepublic', 'active': 1962,
        'notes': 'Sabana de la Mar; early record only 1956-1962',
    },

    # ------------------------------------------------------------------
    # Belize
    # ------------------------------------------------------------------
    'Belize': {
        'wmo_id': '78583', 'lat': 17.53, 'lon': -88.30, 'elev':  5,
        'country': 'Belize', 'active': 2025,
        'notes': 'Belize/Phillip Goldston Intl.; active 1980-2025',
    },

    # ------------------------------------------------------------------
    # Guatemala
    # ------------------------------------------------------------------
    'GuatemalaCity': {
        'wmo_id': '78641', 'lat': 14.53, 'lon': -90.57, 'elev': 1496,
        'country': 'Guatemala', 'active': 1990,
        'notes': 'Guatemala City; record 1973-1990 in IGRA',
    },
    'PuertoBarrios': {
        'wmo_id': '78637', 'lat': 15.72, 'lon': -88.60, 'elev':  1,
        'country': 'Guatemala', 'active': 1978,
        'notes': 'Puerto Barrios, Caribbean coast; short record 1974-1978',
    },
    'SanJose_GT': {
        'wmo_id': '78647', 'lat': 13.92, 'lon': -90.82, 'elev':  2,
        'country': 'Guatemala', 'active': 1978,
        'notes': 'San Jose, Guatemala (Pacific coast); short record 1974-1978',
    },

    # ------------------------------------------------------------------
    # Honduras
    # ------------------------------------------------------------------
    'SwanIsland': {
        'wmo_id': '78501', 'lat': 17.40, 'lon': -83.93, 'elev': 11,
        'country': 'Honduras', 'active': 1980,
        'notes': 'Swan Island (Isla del Cisne), Caribbean; 1948-1980; '
                 'strategically important Gulf location',
    },
    'Tegucigalpa': {
        'wmo_id': '78720', 'lat': 14.05, 'lon': -87.25, 'elev': 1002,
        'country': 'Honduras', 'active': 1997,
        'notes': 'Tegucigalpa; record 1976-1997 in IGRA',
    },
    'SotoCano': {
        'wmo_id': '78721', 'lat': 14.37, 'lon': -87.62, 'elev': 628,
        'country': 'Honduras', 'active': 2012,
        'notes': 'Soto Cano AB (US military); 2008-2012',
    },

    # ------------------------------------------------------------------
    # Nicaragua
    # ------------------------------------------------------------------
    'Managua': {
        'wmo_id': '78741', 'lat': 12.15, 'lon': -86.17, 'elev': 56,
        'country': 'Nicaragua', 'active': 2007,
        'notes': 'Managua A.C. Sandino; long record 1947-2007; intermittent '
                 'operations due to funding/equipment issues',
    },
    'PuertoCabezas': {
        'wmo_id': '78730', 'lat': 14.07, 'lon': -83.37, 'elev': 20,
        'country': 'Nicaragua', 'active': 1983,
        'notes': 'Puerto Cabezas, Caribbean coast; 1973-1983',
    },

    # ------------------------------------------------------------------
    # El Salvador
    # ------------------------------------------------------------------
    'SanSalvador': {
        'wmo_id': '78663', 'lat': 13.70, 'lon': -89.12, 'elev': 621,
        'country': 'ElSalvador', 'active': 1976,
        'notes': 'San Salvador/Ilopango; short record 1973-1976',
    },

    # ------------------------------------------------------------------
    # Costa Rica
    # ------------------------------------------------------------------
    'SanJose_CR': {
        'wmo_id': '78762', 'lat':  9.98, 'lon': -84.18, 'elev': 908,
        'country': 'CostaRica', 'active': 2021,
        'notes': 'Juan Santamaria Intl Airport; active 1972-2021',
    },

    # ------------------------------------------------------------------
    # Panama
    # ------------------------------------------------------------------
    'Panama': {
        'wmo_id': '78807', 'lat':  8.97, 'lon': -79.57, 'elev':  7,
        'country': 'Panama', 'active': 2026,
        'notes': 'Corozal Oeste (was Howard AFB); active 1946-present',
    },

    # ------------------------------------------------------------------
    # Cayman Islands
    # ------------------------------------------------------------------
    'GrandCayman': {
        'wmo_id': '78384', 'lat': 19.29, 'lon': -81.36, 'elev':  3,
        'country': 'CaymanIslands', 'active': 2026,
        'notes': 'Owen Roberts Airport, Grand Cayman; active 1956-present',
    },

    # ------------------------------------------------------------------
    # Bahamas
    # ------------------------------------------------------------------
    'Nassau': {
        'wmo_id': '78073', 'lat': 25.05, 'lon': -77.47, 'elev':  7,
        'country': 'Bahamas', 'active': 2020,
        'notes': 'Nassau Airport, New Providence; 1977-2020',
    },
    'GoldRockCreek': {
        'wmo_id': '78063', 'lat': 26.62, 'lon': -78.37, 'elev':  6,
        'country': 'Bahamas', 'active': 1970,
        'notes': 'Gold Rock Creek (Grand Bahama); historical 1951-1970',
    },
    'Eleuthera': {
        'wmo_id': '78076', 'lat': 25.27, 'lon': -76.30, 'elev': 10,
        'country': 'Bahamas', 'active': 1970,
        'notes': 'Eleuthera Island; historical 1952-1970',
    },
    'TurksIsland': {
        'wmo_id': '78118', 'lat': 21.45, 'lon': -71.15, 'elev': 10,
        'country': 'TurksAndCaicos', 'active': 1978,
        'notes': 'Turks Island; 1954-1978',
    },

    # ------------------------------------------------------------------
    # Puerto Rico / US Caribbean
    # ------------------------------------------------------------------
    'SanJuan': {
        'wmo_id': '78526', 'lat': 18.43, 'lon': -65.99, 'elev':  4,
        'country': 'USA', 'active': 2026,
        'notes': 'San Juan/Int., Puerto Rico; NWS launch site; active 1946-present',
    },

    # ------------------------------------------------------------------
    # Bermuda (just inside lon range)
    # ------------------------------------------------------------------
    'Bermuda': {
        'wmo_id': '78016', 'lat': 32.37, 'lon': -64.68, 'elev':  4,
        'country': 'Bermuda', 'active': 2026,
        'notes': 'L.F. Wade Intl Airport; active 1946-present',
    },

    # ------------------------------------------------------------------
    # Colombia (San Andres Island – Gulf-adjacent Caribbean)
    # ------------------------------------------------------------------
    'SanAndres': {
        'wmo_id': '80001', 'lat': 12.58, 'lon': -81.72, 'elev':  1,
        'country': 'Colombia', 'active': 2026,
        'notes': 'San Andres (Isla) / Sesquicentenario; active 1956-present; '
                 'strategically placed in western Caribbean',
    },

    # ------------------------------------------------------------------
    # Lesser Antilles (eastern edge of domain)
    # ------------------------------------------------------------------
    'Guadeloupe': {
        'wmo_id': '78897', 'lat': 16.26, 'lon': -61.52, 'elev': 11,
        'country': 'France', 'active': 2026,
        'notes': 'Le Raizet Aero, Guadeloupe; active 1952-present',
    },
    'StMaarten': {
        'wmo_id': '78866', 'lat': 18.04, 'lon': -63.12, 'elev':  3,
        'country': 'Netherlands', 'active': 2026,
        'notes': 'Juliana Airport, St. Maarten; active 1956-present',
    },
    'Curacao': {
        'wmo_id': '78988', 'lat': 12.20, 'lon': -68.97, 'elev':  8,
        'country': 'Netherlands', 'active': 2026,
        'notes': 'Hato Airport, Curacao; active 1956-present',
    },
    'Trinidad': {
        'wmo_id': '78970', 'lat': 10.59, 'lon': -61.34, 'elev': 12,
        'country': 'Trinidad', 'active': 2026,
        'notes': 'Piarco Intl Airport, Trinidad; active 1969-present',
    },

}


# ---------------------------------------------------------------------------
# Convenience: subset dictionaries for different subregions
# ---------------------------------------------------------------------------

# Currently-active stations only (last record year >= 2010)
ACTIVE_STATIONS = {
    k: v for k, v in STATION_NAMES.items() if v['active'] >= 2010
}

# US Gulf Coast only
US_GULF_STATIONS = {
    k: v for k, v in STATION_NAMES.items()
    if v['country'] == 'USA' and v['lon'] <= -79.0 and v['lat'] <= 35.0
}

# Mexico Gulf / Yucatan corridor (lon > -105, lat 14-28)
MEXICO_GULF_STATIONS = {
    k: v for k, v in STATION_NAMES.items()
    if v['country'] == 'Mexico' and v['lon'] >= -100 and v['lat'] >= 14
}

# Caribbean basin (lat 8-25, lon -90 to -60)
CARIBBEAN_STATIONS = {
    k: v for k, v in STATION_NAMES.items()
    if 8 <= v['lat'] <= 25 and -90 <= v['lon'] <= -60
    and v['country'] not in ('USA', 'Mexico')
}


if __name__ == '__main__':
    print(f"Total stations in STATION_NAMES: {len(STATION_NAMES)}")
    print(f"Active stations (record >= 2010): {len(ACTIVE_STATIONS)}")
    print(f"US Gulf Coast stations: {len(US_GULF_STATIONS)}")
    print(f"Mexico Gulf/Yucatan stations: {len(MEXICO_GULF_STATIONS)}")
    print(f"Caribbean basin stations: {len(CARIBBEAN_STATIONS)}")
    print()
    print("Active stations:")
    for name, info in sorted(ACTIVE_STATIONS.items()):
        print(f"  {name:20s}  WMO={info['wmo_id']}  "
              f"lat={info['lat']:6.2f}  lon={info['lon']:7.2f}  "
              f"elev={info['elev']:5.0f}m  {info['country']}")
