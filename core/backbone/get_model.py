from core.backbone import __dict__


def get_model(arch, num_classes):
    return __dict__[arch](num_classes=num_classes)
