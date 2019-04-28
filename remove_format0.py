import os

for name in os.listdir('samples'):
    if name.endswith('_format0.mid'):
        os.rename("samples/"+name, "samples/"+''.join(name.split('_format0')))
