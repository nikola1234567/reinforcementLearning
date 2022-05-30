from datetime import datetime
import os

from configurations import NAS_ENVIRONMENT_LOG_DIR

now = datetime.now()
append_mode = 'a'


class NotepadHandler:

    def __init__(self, dataset_name, optional_description=None):
        if not os.path.exists(NAS_ENVIRONMENT_LOG_DIR):
            os.mkdir(NAS_ENVIRONMENT_LOG_DIR)
        file_name = '{}_{}'.format(dataset_name, now.strftime("%m_%d_%Y_%H_%M"))
        self.file_path = os.path.join(NAS_ENVIRONMENT_LOG_DIR, file_name)
        if not os.path.exists(self.file_path):
            os.mkdir(self.file_path)
        self.file_path = os.path.join(self.file_path, '{}.txt'.format(file_name))
        with open(self.file_path, append_mode) as f:
            f.write(file_name)
            f.flush()
        if optional_description is not None:
            self.write(content=optional_description, content_title="Optional description")

    def write(self, content, content_title):
        padding = '\n\n\n\n'
        top_section_border = '======== Section Title ==============================================\n'
        border = '=====================================================================\n'
        with open(self.file_path, append_mode) as f:
            content_title = "".join((content_title, '\n'))
            content = "".join((content, '\n'))
            f.write(padding)
            f.write(top_section_border)
            f.write(content_title)
            f.write(border)
            f.write(content)
            f.write(border)
            f.flush()
