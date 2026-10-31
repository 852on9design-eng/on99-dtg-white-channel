# ON99 DTG White Channel

Streamlit app that turns a transparent PNG into a PrintEXP-ready TIFF with a Photoshop **white** spot channel (DTG/DTF white underbase).

## Features

- White underbase follows alpha (ink gets white; knockout/holes stay empty)
- Choke, optional X offset, bottom fade, crop, lead-in bar, registration guides
- Default export: CMYK + Spot `white` for Hoson PrintEXP

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docs

See [EXPORT.md](EXPORT.md) for PrintEXP TIFF specifics and troubleshooting.

## License

MIT — see [LICENSE](LICENSE).
