# Storage

Placeholder to commit directory and ignore contents.

If you run out of space storing the models, make sure you aren't storing the events files. 
Here is a simple bash command to remove them:

```bash
rm $(find ./runs -name events*)
rm $(find ./storage -name events*)
```