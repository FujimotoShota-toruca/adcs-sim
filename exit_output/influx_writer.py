# influx_writer.py
from influxdb_client import InfluxDBClient, Point, WritePrecision

class InfluxWriter:
    def __init__(self, url, token, org, bucket):
        self._client = InfluxDBClient(url=url, token=token, org=org)
        self._write_api = self._client.write_api(write_options=WritePrecision.S)
        self._bucket = bucket

    def write(self, measurement: str, tags: dict, fields: dict, time):
        pt = Point(measurement).time(time)
        for k,v in tags.items():
            pt.tag(k, v)
        for k,v in fields.items():
            pt.field(k, v)
        self._write_api.write(bucket=self._bucket, record=pt)
