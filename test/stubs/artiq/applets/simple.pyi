class SimpleApplet:
    def __init__(self, main_widget_class, cmd_description=None,
                 default_update_delay=0.0):
        ...

    def add_dataset(self, name, help=None, required=True):
        ...

    def args_init(self):
        ...

    def quamash_init(self):
        ...

    def ipc_init(self):
        ...

    def ipc_close(self):
        ...

    def create_main_widget(self):
        ...

    def sub_init(self, data):
        ...

    def filter_mod(self, mod):
        ...

    def emit_data_changed(self, data, mod_buffer):
        ...

    def flush_mod_buffer(self):
        ...

    def sub_mod(self, mod):
        ...

    def subscribe(self):
        ...

    def unsubscribe(self):
        ...

    def run(self):
        ...


class TitleApplet(SimpleApplet):
    def __init__(self, *args, **kwargs):
        ...

    def args_init(self):
        ...

    def emit_data_changed(self, data, mod_buffer):
        ...