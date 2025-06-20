Dalam dashboard monitoring Grafana, saya memvisualisasikan 6 metrik berbeda dalam bentuk grafik time series. Untuk alasan efisiensi tampilan dan keterbacaan, saya mengelompokkan **3 metrik** sekaligus dalam satu **Pie Chart**.

Oleh karena itu, dalam 2 screenshot panel berbeda, sudah mencakup total 6 metrik, yaitu:

**📊 Panel 1: Model Classification Metrics**
Menampilkan:

`precision_hoax`

`recall_hoax`

`f1_hoax`

**📊 Panel 2: Non-hoax Classification Metrics**
Menampilkan:

`precision_non_hoax`

`recall_non_hoax`

`f1_non_hoax`

Dengan pendekatan ini, semua metrik dapat:

- Dipantau secara bersamaan dalam konteks performa kelas tertentu

- Dibandingkan antar metrik secara visual dalam satu grafik

