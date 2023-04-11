import javaobj


def main():
  with open("./datasets/imdbProfiles", "rb") as fd:
    marshaller = javaobj.JavaObjectUnmarshaller(fd)
    pobj = marshaller.readObject()
    print(pobj)


if __name__=="__main__":
  main()