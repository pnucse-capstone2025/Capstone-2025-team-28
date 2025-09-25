import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom

def generate_single_mpd(output_dir, num_segments, segment_duration=10):
    mpd = ET.Element("MPD", {
        "xmlns": "urn:mpeg:dash:schema:mpd:2011",
        "profiles": "urn:mpeg:dash:profile:isoff-on-demand:2011",
        "type": "static",
        "mediaPresentationDuration": f"PT{num_segments * segment_duration}S",
        "minBufferTime": "PT1.5S"
    })

    period = ET.SubElement(mpd, "Period", {"start": "PT0S"})
    adaptation_set = ET.SubElement(period, "AdaptationSet", {
        "mimeType": "video/mp4",
        "segmentAlignment": "true",
        "startWithSAP": "1"
    })

    resolutions = [
        ("360p", 640, 360, "avc1.42c01e", 500000),
        ("480p", 854, 480, "avc1.4d401f", 800000),
        ("720p", 1280, 720, "avc1.64001f", 1200000),
        ("1080p", 1920, 1080, "avc1.640028", 2500000)
    ]

    for rep_id, (label, width, height, codec, bandwidth) in enumerate(resolutions):
        representation = ET.SubElement(adaptation_set, "Representation", {
            "id": str(rep_id),
            "codecs": codec,
            "width": str(width),
            "height": str(height),
            "bandwidth": str(bandwidth)
        })

        ET.SubElement(representation, "BaseURL").text = "./"

        segment_template = ET.SubElement(representation, "SegmentTemplate", {
            "timescale": "1000",  # ms 단위
            "media": f"chunk-stream{rep_id}-$Number$.m4s",
            "initialization": f"init-stream{rep_id}.mp4",
            "startNumber": "0",
            "duration": str(segment_duration * 1000)  # ms
        })
        
    return mpd

def pretty_print_xml(elem: ET.Element, output_path: str):
    """ElementTree XML을 예쁘게 출력 (들여쓰기 적용)"""
    rough_string = ET.tostring(elem, encoding='utf-8')
    reparsed = minidom.parseString(rough_string)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(reparsed.toprettyxml(indent="  "))

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python generate_single_mpd.py [output_dir] [num_segments]")
        exit(1)

    output_dir = sys.argv[1]
    num_segments = int(sys.argv[2])

    mpd_elem = generate_single_mpd(output_dir, num_segments)
    mpd_path = os.path.join(output_dir, "manifest.mpd")
    pretty_print_xml(mpd_elem, mpd_path)
    print(f"[✅] MPD 파일 생성 완료: {mpd_path}")
    
