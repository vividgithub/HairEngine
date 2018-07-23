#include <iostream>

#include "../geo/hair.h"

int main() {
	using namespace HairEngine;
	using namespace std;

	cout << "Reading Hair" << endl;
	Hair hair("C:\\Users\\VividWinPC1\\Developer\\Project\\HairEngine\\Houdini\\Resources\\Models\\Feamle 04 Retop\\Hair\\Straight-50000.hair");
	cout << "Writing Hair" << endl;
	hair.writeToFile("C:\\Users\\VividWinPC1\\Desktop\\Test.hair");

	cout << "Done" << endl;
	return 0;
}