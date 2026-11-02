"""Visible PNG snapshots with embedded reproducible analysis metadata."""
import json
from dataclasses import asdict


def result_metadata(result, source):
    return dict(version=2,recording=str(source),channel=result.channel,fs=result.fs,
                settings=asdict(result.settings),trials=[asdict(t.trial) for t in result.trials],
                quality=result.quality,filter_mode="offline zero-phase",
                feature_definitions=[asdict(s) for s in result.features.profile.visible_specs()],
                units=dict(signal="uV",fft="uV",psd="uV^2/Hz",power="uV^2"))


def save_png(widget,path,metadata):
    image=widget.grab().toImage()
    image.setText("Analysis",json.dumps(metadata,ensure_ascii=False,allow_nan=False))
    return image.save(str(path),"PNG")
