
This repository compares the time-to-train of two gradient-boosted
decision tree (GBDT) frameworks - LightGBM and XGBoost - across
different hardware and versions. The purpose of this comparison is to
help the CCAO make two decisions:

1.  Which GBDT framework to use for its 2024 automated valuation models
2.  Whether or not to purchase additional hardware (a GPU) in order to
    improve model training speed

Below are the results of our tests. ***Please note that the performance
statistics presented here are only for cross-model comparison and do not
reflect any real model results.*** They are basically included to show
that each framework and version generates similar results, given the
same data.

<div id="spigpllede" style="padding-left:0px;padding-right:0px;padding-top:10px;padding-bottom:10px;overflow-x:auto;overflow-y:auto;width:auto;height:auto;">
<style>#spigpllede table {
  font-family: system-ui, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol', 'Noto Color Emoji';
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

#spigpllede thead, #spigpllede tbody, #spigpllede tfoot, #spigpllede tr, #spigpllede td, #spigpllede th {
  border-style: none;
}

#spigpllede p {
  margin: 0;
  padding: 0;
}

#spigpllede .gt_table {
  display: table;
  border-collapse: collapse;
  line-height: normal;
  margin-left: auto;
  margin-right: auto;
  color: #333333;
  font-size: 16px;
  font-weight: normal;
  font-style: normal;
  background-color: #FFFFFF;
  width: auto;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #A8A8A8;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #A8A8A8;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
}

#spigpllede .gt_caption {
  padding-top: 4px;
  padding-bottom: 4px;
}

#spigpllede .gt_title {
  color: #333333;
  font-size: 125%;
  font-weight: initial;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-color: #FFFFFF;
  border-bottom-width: 0;
}

#spigpllede .gt_subtitle {
  color: #333333;
  font-size: 85%;
  font-weight: initial;
  padding-top: 3px;
  padding-bottom: 5px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-color: #FFFFFF;
  border-top-width: 0;
}

#spigpllede .gt_heading {
  background-color: #FFFFFF;
  text-align: center;
  border-bottom-color: #FFFFFF;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#spigpllede .gt_bottom_border {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#spigpllede .gt_col_headings {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
}

#spigpllede .gt_col_heading {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 6px;
  padding-left: 5px;
  padding-right: 5px;
  overflow-x: hidden;
}

#spigpllede .gt_column_spanner_outer {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: normal;
  text-transform: inherit;
  padding-top: 0;
  padding-bottom: 0;
  padding-left: 4px;
  padding-right: 4px;
}

#spigpllede .gt_column_spanner_outer:first-child {
  padding-left: 0;
}

#spigpllede .gt_column_spanner_outer:last-child {
  padding-right: 0;
}

#spigpllede .gt_column_spanner {
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: bottom;
  padding-top: 5px;
  padding-bottom: 5px;
  overflow-x: hidden;
  display: inline-block;
  width: 100%;
}

#spigpllede .gt_spanner_row {
  border-bottom-style: hidden;
}

#spigpllede .gt_group_heading {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  text-align: left;
}

#spigpllede .gt_empty_group_heading {
  padding: 0.5px;
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  vertical-align: middle;
}

#spigpllede .gt_from_md > :first-child {
  margin-top: 0;
}

#spigpllede .gt_from_md > :last-child {
  margin-bottom: 0;
}

#spigpllede .gt_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  margin: 10px;
  border-top-style: solid;
  border-top-width: 1px;
  border-top-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 1px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 1px;
  border-right-color: #D3D3D3;
  vertical-align: middle;
  overflow-x: hidden;
}

#spigpllede .gt_stub {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
}

#spigpllede .gt_stub_row_group {
  color: #333333;
  background-color: #FFFFFF;
  font-size: 100%;
  font-weight: initial;
  text-transform: inherit;
  border-right-style: solid;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
  padding-left: 5px;
  padding-right: 5px;
  vertical-align: top;
}

#spigpllede .gt_row_group_first td {
  border-top-width: 2px;
}

#spigpllede .gt_row_group_first th {
  border-top-width: 2px;
}

#spigpllede .gt_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#spigpllede .gt_first_summary_row {
  border-top-style: solid;
  border-top-color: #D3D3D3;
}

#spigpllede .gt_first_summary_row.thick {
  border-top-width: 2px;
}

#spigpllede .gt_last_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#spigpllede .gt_grand_summary_row {
  color: #333333;
  background-color: #FFFFFF;
  text-transform: inherit;
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
}

#spigpllede .gt_first_grand_summary_row {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-top-style: double;
  border-top-width: 6px;
  border-top-color: #D3D3D3;
}

#spigpllede .gt_last_grand_summary_row_top {
  padding-top: 8px;
  padding-bottom: 8px;
  padding-left: 5px;
  padding-right: 5px;
  border-bottom-style: double;
  border-bottom-width: 6px;
  border-bottom-color: #D3D3D3;
}

#spigpllede .gt_striped {
  background-color: rgba(128, 128, 128, 0.05);
}

#spigpllede .gt_table_body {
  border-top-style: solid;
  border-top-width: 2px;
  border-top-color: #D3D3D3;
  border-bottom-style: solid;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
}

#spigpllede .gt_footnotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#spigpllede .gt_footnote {
  margin: 0px;
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#spigpllede .gt_sourcenotes {
  color: #333333;
  background-color: #FFFFFF;
  border-bottom-style: none;
  border-bottom-width: 2px;
  border-bottom-color: #D3D3D3;
  border-left-style: none;
  border-left-width: 2px;
  border-left-color: #D3D3D3;
  border-right-style: none;
  border-right-width: 2px;
  border-right-color: #D3D3D3;
}

#spigpllede .gt_sourcenote {
  font-size: 90%;
  padding-top: 4px;
  padding-bottom: 4px;
  padding-left: 5px;
  padding-right: 5px;
}

#spigpllede .gt_left {
  text-align: left;
}

#spigpllede .gt_center {
  text-align: center;
}

#spigpllede .gt_right {
  text-align: right;
  font-variant-numeric: tabular-nums;
}

#spigpllede .gt_font_normal {
  font-weight: normal;
}

#spigpllede .gt_font_bold {
  font-weight: bold;
}

#spigpllede .gt_font_italic {
  font-style: italic;
}

#spigpllede .gt_super {
  font-size: 65%;
}

#spigpllede .gt_footnote_marks {
  font-size: 75%;
  vertical-align: 0.4em;
  position: initial;
}

#spigpllede .gt_asterisk {
  font-size: 100%;
  vertical-align: 0;
}

#spigpllede .gt_indent_1 {
  text-indent: 5px;
}

#spigpllede .gt_indent_2 {
  text-indent: 10px;
}

#spigpllede .gt_indent_3 {
  text-indent: 15px;
}

#spigpllede .gt_indent_4 {
  text-indent: 20px;
}

#spigpllede .gt_indent_5 {
  text-indent: 25px;
}
</style>
<table class="gt_table" data-quarto-disable-processing="false" data-quarto-bootstrap="false">
  <thead>
    <tr class="gt_heading">
      <td colspan="17" class="gt_heading gt_title gt_font_normal gt_bottom_border" style="font-weight: bold;">LightGBM CPU/GPU vs XGBoost CPU/GPU</td>
    </tr>
    
    <tr class="gt_col_headings">
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Server">Server</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Type">Type</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Device">Device</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Package Version">Package Version</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_left" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Hardware Specifications">Hardware Specifications</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Approx. Peak Device Utilization">Approx. Peak Device Utilization</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Wall Time (Full Run)">Wall Time (Full Run)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Wall Time (Prediction)">Wall Time (Prediction)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="Wall Time (SHAP)">Wall Time (SHAP)</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="RMSE">RMSE</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="MAE">MAE</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="MAPE">MAPE</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="R2">R2</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="COD">COD</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="PRD">PRD</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="PRB">PRB</th>
      <th class="gt_col_heading gt_columns_bottom_border gt_right" rowspan="1" colspan="1" style="font-weight: bold;" scope="col" id="MKI">MKI</th>
    </tr>
  </thead>
  <tbody class="gt_table_body">
    <tr><td headers="Server" class="gt_row gt_left">CCAO</td>
<td headers="Type" class="gt_row gt_left">LightGBM</td>
<td headers="Device" class="gt_row gt_left">CPU</td>
<td headers="Package Version" class="gt_row gt_right">3.3.5</td>
<td headers="Hardware Specifications" class="gt_row gt_left">Xeon Silver 4208 CPU @ 2.10GHz, 16 cores</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">100%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">3m 55s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">1m 5s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">2h 1m 38s</td>
<td headers="RMSE" class="gt_row gt_right">$130,989</td>
<td headers="MAE" class="gt_row gt_right">$74,124</td>
<td headers="MAPE" class="gt_row gt_right">26.97%</td>
<td headers="R2" class="gt_row gt_right">0.883</td>
<td headers="COD" class="gt_row gt_right">27.634</td>
<td headers="PRD" class="gt_row gt_right">1.140</td>
<td headers="PRB" class="gt_row gt_right">−0.225</td>
<td headers="MKI" class="gt_row gt_right">0.851</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">CCAO</td>
<td headers="Type" class="gt_row gt_left">LightGBM</td>
<td headers="Device" class="gt_row gt_left">CPU</td>
<td headers="Package Version" class="gt_row gt_right">4.1.0</td>
<td headers="Hardware Specifications" class="gt_row gt_left">Xeon Silver 4208 CPU @ 2.10GHz, 16 cores</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">100%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">4m 8s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">1m 4s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">2h 1m 48s</td>
<td headers="RMSE" class="gt_row gt_right">$130,989</td>
<td headers="MAE" class="gt_row gt_right">$74,124</td>
<td headers="MAPE" class="gt_row gt_right">26.97%</td>
<td headers="R2" class="gt_row gt_right">0.883</td>
<td headers="COD" class="gt_row gt_right">27.634</td>
<td headers="PRD" class="gt_row gt_right">1.140</td>
<td headers="PRB" class="gt_row gt_right">−0.225</td>
<td headers="MKI" class="gt_row gt_right">0.851</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">CCAO</td>
<td headers="Type" class="gt_row gt_left">XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></td>
<td headers="Device" class="gt_row gt_left">CPU</td>
<td headers="Package Version" class="gt_row gt_right">2.0.0.1</td>
<td headers="Hardware Specifications" class="gt_row gt_left">Xeon Silver 4208 CPU @ 2.10GHz, 16 cores</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">100%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">2m 41s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">12s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">2m 14s</td>
<td headers="RMSE" class="gt_row gt_right">$126,389</td>
<td headers="MAE" class="gt_row gt_right">$74,129</td>
<td headers="MAPE" class="gt_row gt_right">26.76%</td>
<td headers="R2" class="gt_row gt_right">0.885</td>
<td headers="COD" class="gt_row gt_right">27.458</td>
<td headers="PRD" class="gt_row gt_right">1.130</td>
<td headers="PRB" class="gt_row gt_right">−0.212</td>
<td headers="MKI" class="gt_row gt_right">0.866</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">NVIDIA</td>
<td headers="Type" class="gt_row gt_left">LightGBM</td>
<td headers="Device" class="gt_row gt_left">CPU</td>
<td headers="Package Version" class="gt_row gt_right">3.3.5</td>
<td headers="Hardware Specifications" class="gt_row gt_left">Xeon Gold 6354 CPU @ 3.00GHz, 72 cores</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">100%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">4m 12s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">9s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">19m 5s</td>
<td headers="RMSE" class="gt_row gt_right">$130,989</td>
<td headers="MAE" class="gt_row gt_right">$74,124</td>
<td headers="MAPE" class="gt_row gt_right">26.97%</td>
<td headers="R2" class="gt_row gt_right">0.883</td>
<td headers="COD" class="gt_row gt_right">27.634</td>
<td headers="PRD" class="gt_row gt_right">1.140</td>
<td headers="PRB" class="gt_row gt_right">−0.225</td>
<td headers="MKI" class="gt_row gt_right">0.851</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">NVIDIA</td>
<td headers="Type" class="gt_row gt_left">LightGBM</td>
<td headers="Device" class="gt_row gt_left">CPU</td>
<td headers="Package Version" class="gt_row gt_right">4.1.0.99</td>
<td headers="Hardware Specifications" class="gt_row gt_left">Xeon Gold 6354 CPU @ 3.00GHz, 72 cores</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">100%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">4m 7s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">9s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">18m 52s</td>
<td headers="RMSE" class="gt_row gt_right">$130,989</td>
<td headers="MAE" class="gt_row gt_right">$74,124</td>
<td headers="MAPE" class="gt_row gt_right">26.97%</td>
<td headers="R2" class="gt_row gt_right">0.883</td>
<td headers="COD" class="gt_row gt_right">27.634</td>
<td headers="PRD" class="gt_row gt_right">1.140</td>
<td headers="PRB" class="gt_row gt_right">−0.225</td>
<td headers="MKI" class="gt_row gt_right">0.851</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">NVIDIA</td>
<td headers="Type" class="gt_row gt_left">LightGBM<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span></td>
<td headers="Device" class="gt_row gt_left">GPU</td>
<td headers="Package Version" class="gt_row gt_right">4.1.0.99</td>
<td headers="Hardware Specifications" class="gt_row gt_left">NVIDIA A40, 48GB</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">10%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">5m 56s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">11s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">19m 10s</td>
<td headers="RMSE" class="gt_row gt_right">$130,546</td>
<td headers="MAE" class="gt_row gt_right">$74,296</td>
<td headers="MAPE" class="gt_row gt_right">27.27%</td>
<td headers="R2" class="gt_row gt_right">0.884</td>
<td headers="COD" class="gt_row gt_right">27.929</td>
<td headers="PRD" class="gt_row gt_right">1.143</td>
<td headers="PRB" class="gt_row gt_right">−0.231</td>
<td headers="MKI" class="gt_row gt_right">0.846</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">NVIDIA</td>
<td headers="Type" class="gt_row gt_left">XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></td>
<td headers="Device" class="gt_row gt_left">CPU</td>
<td headers="Package Version" class="gt_row gt_right">2.0.0.1</td>
<td headers="Hardware Specifications" class="gt_row gt_left">Xeon Gold 6354 CPU @ 3.00GHz, 72 cores</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">100%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">1m 6s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">4s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">20s</td>
<td headers="RMSE" class="gt_row gt_right">$126,024</td>
<td headers="MAE" class="gt_row gt_right">$73,961</td>
<td headers="MAPE" class="gt_row gt_right">26.76%</td>
<td headers="R2" class="gt_row gt_right">0.886</td>
<td headers="COD" class="gt_row gt_right">27.467</td>
<td headers="PRD" class="gt_row gt_right">1.130</td>
<td headers="PRB" class="gt_row gt_right">−0.209</td>
<td headers="MKI" class="gt_row gt_right">0.867</td></tr>
    <tr><td headers="Server" class="gt_row gt_left">NVIDIA</td>
<td headers="Type" class="gt_row gt_left">XGBoost<span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span></td>
<td headers="Device" class="gt_row gt_left">GPU</td>
<td headers="Package Version" class="gt_row gt_right">2.0.0.1</td>
<td headers="Hardware Specifications" class="gt_row gt_left">NVIDIA A40, 48GB</td>
<td headers="Approx. Peak Device Utilization" class="gt_row gt_right">92%</td>
<td headers="Wall Time (Full Run)" class="gt_row gt_right">1m 11s</td>
<td headers="Wall Time (Prediction)" class="gt_row gt_right">13s</td>
<td headers="Wall Time (SHAP)" class="gt_row gt_right">4s</td>
<td headers="RMSE" class="gt_row gt_right">$126,366</td>
<td headers="MAE" class="gt_row gt_right">$73,981</td>
<td headers="MAPE" class="gt_row gt_right">26.86%</td>
<td headers="R2" class="gt_row gt_right">0.885</td>
<td headers="COD" class="gt_row gt_right">27.543</td>
<td headers="PRD" class="gt_row gt_right">1.131</td>
<td headers="PRB" class="gt_row gt_right">−0.214</td>
<td headers="MKI" class="gt_row gt_right">0.865</td></tr>
  </tbody>
  
  <tfoot class="gt_footnotes">
    <tr>
      <td class="gt_footnote" colspan="17"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>1</sup></span> Categoricals with over 50 values are hashed, otherwise one-hot encoded</td>
    </tr>
    <tr>
      <td class="gt_footnote" colspan="17"><span class="gt_footnote_marks" style="white-space:nowrap;font-style:italic;font-weight:normal;"><sup>2</sup></span> Categoricals with over 50 values are hashed, otherwise native (un touched)</td>
    </tr>
  </tfoot>
</table>
</div>

- Comparing hardware

- Model switch decision

- Purchasing decision

- populate input data

- Show params

- Tasks

  - Train using N sales, predict on test (perf numbers)
  - Time to train full set
  - Time to predict N rows
  - Time to calc N shaps

- Time savings calculation
