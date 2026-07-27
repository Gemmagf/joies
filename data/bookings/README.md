# Hotel Bookings dataset

`hotel_bookings.csv.gz` (~1.4 MB gzipped, ~19 MB raw, 118,547 rows).

**Source.** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand
datasets.* Data in Brief, 22, 41–49. <https://doi.org/10.1016/j.dib.2018.11.126>.
Mirrored on Kaggle as [`jessemostipak/hotel-booking-demand`](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).

**License.** CC BY 4.0 (per the Data in Brief publication).

**What it is.** Two years of real booking records (2015-07 → 2017-08) from one
city hotel and one resort hotel in Portugal, released for research. Guest-level
identifiers have been removed by the authors.

**How we use it.**

- Ground truth for the Profile & Analytics agent: cancellation risk (churn),
  single-booking revenue (CLV proxy), and stay-type segmentation.
- **We do not treat any row as a real, identifiable person.** The synthetic
  "Maison" guest personas surfaced in the demo are generated from the aggregate
  distributions of this dataset — realistic, but not any real guest.

**Data quality notes** (applied by [`analytics.dataset`](../../src/maison_concierge/analytics/dataset.py)):

- `children` has 4 missing values → filled with 0.
- `country` has ~490 missing values → filled with `UNK`.
- `agent`, `company` are mostly missing by design → kept as-is, not featured.
- `adr` has a small number of negative values and one large outlier (~5,400) →
  clipped to `[0, 1000]`.
- Rows with `adr == 0` and `is_canceled == 0` (walk-outs / comp stays) are kept
  but flagged.
