from pylabel import importer


dataset = importer.ImportVOC(path="yolov7/swoosh_data/images/val")
dataset.export.ExportToYoloV5("swoosh_data/labels/val")